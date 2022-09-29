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
