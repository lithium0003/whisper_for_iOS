# whisper_for_iOS
iOS implementation of
OpenAI Whisper
https://github.com/openai/whisper

<img src="https://user-images.githubusercontent.com/4783887/192198650-f8d8b2cc-7c96-4a12-a070-198526e2b0ea.png" width="320">

# Limitation 
On iPhone13 Pro, model size over "medium" is too big for Neural Engine. So, you should use under "small" model.

# How to compile
## convert model from pytorch to coreml
As following to original document, install whisper.

In coremltools conversion pytorch -> coreml, error happened here[https://github.com/openai/whisper/blob/5d8d3e75a4826fe5f01205d81c3017a805fc2bf9/whisper/model.py#L192],
"numpy_t op not found".

So, patched version whisper https://github.com/lithium0003/whisper available.

Then, run this file,
``` bash
python convert.py
```

Successfully run, "encoder.mlpackage" and "decoder.mlpackage" will be generated.
Copy these dir to whisper/whisper folder.
``` bash
cp -r *.mlpackage whisper/whisper/
```

## compile on Xcode
Open whisper/whisper.xcodeproj, compile and run.

Enjoy!
