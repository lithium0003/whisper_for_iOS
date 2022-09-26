import numpy as np
import torch
import whisper
import coremltools as ct

def convert_encoder(size="base"):
    model = whisper.load_model(size)
    print(model.dims)

    mel = np.ones([model.dims.n_mels,3000], dtype=np.float32) * -1.5
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

def convert_decoder(size="base"):
    model = whisper.load_model(size)
    print(model.dims)

    audio_features = np.zeros([1,model.dims.n_audio_ctx,model.dims.n_audio_state], dtype=np.float32)
    audio_features = torch.from_numpy(audio_features)
    input_shape = ct.Shape(shape=(1, ct.RangeDim(1, model.dims.n_text_ctx)))
    tokens = np.zeros([1,model.dims.n_text_ctx], dtype=np.int32)
    tokens = torch.from_numpy(tokens)

    traced_model = torch.jit.trace(model.decoder, (tokens,audio_features))

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=input_shape, dtype=int, name="tokens"),
            ct.TensorType(shape=audio_features.shape, name="audio_features"),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
    )
    model.save("decoder.mlpackage")

if __name__=="__main__":
    size = "small"
    convert_encoder(size)
    convert_decoder(size)
