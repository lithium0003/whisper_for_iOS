#!/usr/bin/env python3
import numpy as np
import torch
import whisper
from whisper.model import MultiHeadAttention
import coremltools as ct

def patch_forward(self, x, xa, kv_cache):
    """
    x : torch.LongTensor, shape = (batch_size, <= n_ctx)
        the text tokens
    xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
        the encoded audio features to be attended on
    """
    x = self.token_embedding(x) + self.positional_embedding[self.offset : self.offset + x.shape[1],:]
    x = x.to(xa.dtype)

    for block in self.blocks:
        x = block(x, xa, kv_cache=kv_cache)

    x = self.ln(x)
    logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

    return logits


def convert_encoder(size="base"):
    model = whisper.load_model(size, device='cpu')
    model.eval()
    print(model.dims)

    mel = np.ones([model.dims.n_mels,whisper.audio.N_FRAMES], dtype=np.float32) * -1.5
    input_mel = torch.from_numpy(mel)
    input_mel = input_mel.unsqueeze(0)
    
    traced_model = torch.jit.trace(model.encoder, input_mel)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_mel.shape, dtype=np.float32, name="mel")],
        outputs=[ct.TensorType(dtype=np.float32, name="audio_features")],
        minimum_deployment_target=ct.target.iOS16,
    )
    model.save("encoder.mlpackage")

def convert_initcache(size="base"):
    model = whisper.load_model(size, device='cpu')
    model.eval()
    print(model.dims)

    input_encoder_output = torch.ones((model.dims.n_audio_ctx,model.dims.n_audio_state), dtype=torch.float32)
    input_encoder_output = input_encoder_output.unsqueeze(0)

    class CacheInit(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.decoder
            self.cache, self.hooks = self.install_kv_cache_hooks()

        def forward(self, encoder_output):
            tokens = torch.ones((encoder_output.size()[0],model.dims.n_text_ctx), dtype=torch.int32)
            logits = self.decoder(tokens,encoder_output)
            cross_attn_kvcache = [self.cache[block.cross_attn.key] for block in self.decoder.blocks] + [self.cache[block.cross_attn.value] for block in self.decoder.blocks]
            cross_attn_kvcache = torch.cat(cross_attn_kvcache, dim=0)
            return cross_attn_kvcache

        def install_kv_cache_hooks(self):
            """
            The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
            tensors calculated for the previous positions. This method returns a dictionary that stores
            all caches, and the necessary hooks for the key and value projection modules that save the
            intermediate tensors to be reused during later calculations.

            Returns
            -------
            cache : Dict[nn.Module, torch.Tensor]
                A dictionary object mapping the key/value projection modules to its cache
            hooks : List[RemovableHandle]
                List of PyTorch RemovableHandle objects to stop the hooks to be called
            """
            cache = {}
            hooks = []

            def save_to_cache(module, _, output):
                cache[module] = output
                return cache[module]

            def install_hooks(layer: torch.nn.Module):
                if isinstance(layer, MultiHeadAttention):
                    hooks.append(layer.key.register_forward_hook(save_to_cache))
                    hooks.append(layer.value.register_forward_hook(save_to_cache))

            self.decoder.apply(install_hooks)
            return cache, hooks

    cache_init = CacheInit(model)
    cache_init.eval()

    traced_model = torch.jit.trace(cache_init, input_encoder_output)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=input_encoder_output.shape, dtype=np.float32, name="audio_features"),
        ],
        outputs=[ct.TensorType(dtype=np.float32, name="cross_attn_kvcache")],
        minimum_deployment_target=ct.target.iOS16,
    )
    model.save("decoderinit.mlpackage")

def convert_decoder(size="base"):
    model = whisper.load_model(size, device='cpu')
    model.eval()
    print(model.dims)

    input_encoder_output = torch.ones((model.dims.n_audio_ctx,model.dims.n_audio_state), dtype=torch.float32)
    input_encoder_output = input_encoder_output.unsqueeze(0)
    input_token = torch.ones((1,), dtype=torch.int32)
    input_token = input_token.unsqueeze(0)
    input_cross_attn_kvcache = torch.zeros((model.dims.n_text_layer*2,model.dims.n_audio_ctx,model.dims.n_text_state), dtype=torch.float32)
    input_attn_kvcache = torch.zeros((model.dims.n_text_layer*2,model.dims.n_text_ctx,model.dims.n_text_state), dtype=torch.float32)
    input_offset = torch.tensor([0], dtype=torch.int32)

    class CacheDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()

            bound_method = patch_forward.__get__(model.decoder, model.decoder.__class__)
            setattr(model.decoder, 'forward', bound_method)

            self.decoder = model.decoder

            self.cache = {}
            self.hooks = self.install_kv_cache_hooks()

        def forward(self, tokens, offset, encoder_output, cross_attn_kvcache, attn_kvcache):
            b = len(self.decoder.blocks)
            for i, block in enumerate(self.decoder.blocks):
                self.cache[block.attn.key] = attn_kvcache[i:i+1,:,:]
                self.cache[block.attn.value] = attn_kvcache[i+b:i+b+1,:,:]
                self.cache[block.cross_attn.key] = cross_attn_kvcache[i:i+1,:,:]
                self.cache[block.cross_attn.value] = cross_attn_kvcache[i+b:i+b+1,:,:]
            
            self.decoder.offset = offset

            logits = self.decoder(tokens,encoder_output,self.cache)

            attn_kvcache = [self.cache[block.attn.key] for block in self.decoder.blocks] + [self.cache[block.attn.value] for block in self.decoder.blocks]
            attn_kvcache = torch.cat(attn_kvcache, dim=0)
            return logits, attn_kvcache

        def install_kv_cache_hooks(self):
            """
            The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
            tensors calculated for the previous positions. This method returns a dictionary that stores
            all caches, and the necessary hooks for the key and value projection modules that save the
            intermediate tensors to be reused during later calculations.

            Returns
            -------
            cache : Dict[nn.Module, torch.Tensor]
                A dictionary object mapping the key/value projection modules to its cache
            hooks : List[RemovableHandle]
                List of PyTorch RemovableHandle objects to stop the hooks to be called
            """
            hooks = []

            def save_to_cache(module, _, output):
                if module not in self.cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                    return self.cache[module]
                else:
                    self.cache[module] = torch.cat([self.cache[module], output], dim=1)
                    return self.cache[module]

            def install_hooks(layer: torch.nn.Module):
                if isinstance(layer, MultiHeadAttention):
                    hooks.append(layer.key.register_forward_hook(save_to_cache))
                    hooks.append(layer.value.register_forward_hook(save_to_cache))

            self.decoder.apply(install_hooks)
            return hooks


    cache_decoder = CacheDecoder(model)
    cache_decoder.eval()

    traced_model = torch.jit.trace(cache_decoder, [input_token, input_offset, input_encoder_output, input_cross_attn_kvcache, input_attn_kvcache])

    # print(traced_model)
    # print(traced_model.graph)
    # import logging
    # logging.basicConfig(filename='debug.log', level=logging.DEBUG)

    mlmodel_decoder = ct.convert(traced_model, 
                    convert_to="mlprogram",
                    inputs=[
                        ct.TensorType(shape=input_token.shape, dtype=np.int32, name="tokens"),
                        ct.TensorType(shape=input_offset.shape, dtype=np.int32, name="offset"),
                        ct.TensorType(shape=input_encoder_output.shape, dtype=np.float32, name="audio_features"),
                        ct.TensorType(shape=input_cross_attn_kvcache.shape, dtype=np.float32, name="cross_attn_kvcache"),
                        ct.TensorType(shape=ct.Shape([model.dims.n_text_layer*2,ct.RangeDim(0,model.dims.n_text_ctx),model.dims.n_text_state]), dtype=np.float32, name="attn_kvcache"),
                    ],
                    outputs=[
                        ct.TensorType(dtype=np.float32, name="logits"),
                        ct.TensorType(dtype=np.float32, name="out_attn_kvcache"),
                    ],
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.iOS16)
    mlmodel_decoder.save("decoder.mlpackage")

def testmodel():
    mlmodel_encoder = ct.models.MLModel('encoder.mlpackage')
    mlmodel_initcache = ct.models.MLModel('decoderinit.mlpackage')
    mlmodel_decoder = ct.models.MLModel('decoder.mlpackage')

    mel_shape = mlmodel_encoder.get_spec().description.input[0].type.multiArrayType.shape
    mel = np.ones(mel_shape, dtype=np.float32) * -1.5
    print('encoder')
    out1 = mlmodel_encoder.predict({ 'mel': mel })
    print('initcache')
    out2 = mlmodel_initcache.predict({ **out1 })
    print('decoder')
    attn_kvcache_shape = mlmodel_decoder.get_spec().description.input[-1].type.multiArrayType.shape
    attn_kvcache = np.zeros(attn_kvcache_shape, dtype=np.float32)
    start_tokens = [50258,50259,50359]
    tokens = np.zeros([1,1], dtype=np.int32)
    offset = np.zeros([1], dtype=np.int32)
    for st in start_tokens:
        tokens[0] = st
        out3 = mlmodel_decoder.predict({ 
            'tokens': tokens,
            'offset': offset,
            **out1,
            **out2,
            'attn_kvcache': attn_kvcache,
        })
        attn_kvcache = out3['out_attn_kvcache']
        prob = torch.softmax(torch.from_numpy(out3['logits']), dim=-1).numpy()
        t = np.argmax(out3['logits'],axis=-1)[0,-1]
        print(t, prob[0,-1,t], out3['logits'][0,-1,t])
        if offset[0] == 0:
            print('nospeech', prob[0,0,50362], out3['logits'][0,0,50362])
        offset[0] += 1
    i = 3
    while t != 50257 and i < 448:
        tokens = np.zeros([1,1], dtype=np.int32)
        tokens[0,0] = t
        offset[0] = i
        out3 = mlmodel_decoder.predict({ 
            'tokens': tokens,
            'offset': offset,
            **out1,
            **out2,
            'attn_kvcache': attn_kvcache,
        })
        prob = torch.softmax(torch.from_numpy(out3['logits']), dim=-1).numpy()
        t = np.argmax(out3['logits'],axis=-1)[0,0]
        print(t, prob[0,0,t], out3['logits'][0,0,t])
        i += 1
        #print(out3)


if __name__=="__main__":
    import sys
    sizes = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large']
    
    argv = sys.argv

    if '--test' in argv:
        test = True
        argv.pop(argv.index('--test'))
    else:
        test = False

    if len(argv) < 2:
        size = "base"
    elif argv[1] in sizes:
        size = argv[1]
    else:
        print('available size:',sizes)
        exit()
    
    if test:
        testmodel()
    else:
        convert_encoder(size)
        convert_initcache(size)
        convert_decoder(size)
