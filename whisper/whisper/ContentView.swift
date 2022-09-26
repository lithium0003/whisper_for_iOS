//
//  ContentView.swift
//  whisper
//
//  Created by rei8 on 2022/09/24.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var whisper = WhisperModel.shared
    @State var lang_sele = ""
    @State var trans_sele = 0
    
    var body: some View {
        VStack {
            Spacer()
            HStack {
                Picker(selection: $lang_sele, label: Text("language")) {
                    Text("Auto").tag("")
                    ForEach(whisper.tokenizer.language_codes, id: \.self) { lang in
                        Text(lang)
                    }
                }
                .padding()
                .onChange(of: lang_sele) { lang in
                    whisper.set_lang = lang
                }

                Picker(selection: $trans_sele, label: Text("translate")) {
                    Text("Transcribe").tag(0)
                    Text("To English").tag(1)
                }
                .padding()
                .onChange(of: trans_sele) { trans in
                    if trans == 0 {
                        whisper.transcribe = true
                    }
                    else {
                        whisper.transcribe = false
                    }
                }
            }
            whisper.view()
            Spacer()
            HStack {
                Spacer()
                Button(action: {
                    whisper.start()
                }) {
                    Text("Start")
                }
                .padding()
                Spacer()
                Button(action: {
                    whisper.stop()
                }) {
                    Text("Stop")
                }
                .padding()
                Spacer()
            }
            Spacer()
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
