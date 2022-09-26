//
//  Tokens.swift
//  whisper
//
//  Created by rei8 on 2022/09/25.
//

import Foundation
import CoreML

class Tokenizer {
    let vocab: [String: Int]
    let decoder: [String]
    let codemap: [Character: Int]

    init() {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "json") else {
            vocab = [:]
            decoder = []
            codemap = [:]
            return
        }
        guard let data = try? Data(contentsOf: url) else {
            vocab = [:]
            decoder = []
            codemap = [:]
            return
        }
        guard let vocab = try? JSONDecoder().decode(Dictionary<String,Int>.self, from: data) else {
            self.vocab = [:]
            decoder = []
            codemap = [:]
            return
        }
        self.vocab = vocab
        var reverce_map: [Int: String] = [:]
        for (k,v) in vocab {
            reverce_map[v] = k
        }
        decoder = reverce_map.sorted(by: { $0.key < $1.key }).map({ $0.value })
        
        var bs = [("!", "~"), ("¡", "¬"), ("®", "ÿ")].map({
            let st = $0.0.unicodeScalars.first?.value ?? 0
            let ed = $0.1.unicodeScalars.first?.value ?? 0
            return (st...ed).map({ $0 })
        }).reduce([UInt32](), { $0 + $1 })
        var cs = bs
        var n: UInt32 = 0
        for b: UInt32 in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }
        codemap = Dictionary(uniqueKeysWithValues: zip(cs, bs).map{ (Character( UnicodeScalar($0) ?? " "), Int($1)) })
    }
    
    let eot = 50257
    let sot = 50258
    let sot_translate = 50258
    let sot_transcribe = 50259
    let sot_lm = 50360
    let sot_prev = 50361
    let no_speech = 50362
    let no_timestamps = 50363
    let timestamp_begin = 50364
    
    let language_codes: [String] =
        ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
         "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
         "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
         "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
         "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
         "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
         "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]
    
    
    let language_tokens: [Int] =
        [50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266, 50267, 50268, 50269, 50270, 50271,
         50272, 50273, 50274, 50275, 50276, 50277, 50278, 50279, 50280, 50281, 50282, 50283, 50284,
         50285, 50286, 50287, 50288, 50289, 50290, 50291, 50292, 50293, 50294, 50295, 50296, 50297,
         50298, 50299, 50300, 50301, 50302, 50303, 50304, 50305, 50306, 50307, 50308, 50309, 50310,
         50311, 50312, 50313, 50314, 50315, 50316, 50317, 50318, 50319, 50320, 50321, 50322, 50323,
         50324, 50325, 50326, 50327, 50328, 50329, 50330, 50331, 50332, 50333, 50334, 50335, 50336,
         50337, 50338, 50339, 50340, 50341, 50342, 50343, 50344, 50345, 50346, 50347, 50348, 50349,
         50350, 50351, 50352, 50353, 50354, 50355, 50356, 50357]
    
    func getLanguageToken(lang: String) -> Int {
        language_tokens[language_codes.firstIndex(of: lang) ?? 0]
    }
    
    func getProbLanguage(logits: MLShapedArray<Float>, t_idx: Int = 0) -> [String: Float] {
        var expsum: Float = 0
        var logit_value: [Int: Float] = [:]
        logits.withUnsafeShapedBufferPointer { ptr, shape, stride in
            assert(t_idx < shape[1])
            for lang in language_tokens {
                logit_value[lang] = ptr[stride[1]*t_idx + lang]
            }
        }
        for (_, value) in logit_value {
            expsum += exp(value)
        }
        return Dictionary(uniqueKeysWithValues: logit_value.map({
            (language_codes[language_tokens.firstIndex(of: $0.key)!],
             exp($0.value) / expsum
            )
        }))
    }
    
    func getNoSpeechProb(logits: MLShapedArray<Float>, t_idx: Int = 0) -> Float {
        logits.withUnsafeShapedBufferPointer { ptr, shape, stride in
            var expsum: Float = 0
            assert(t_idx < shape[1])
            for i in 0..<shape[2] {
                let value = ptr[stride[1]*t_idx + i]
                expsum += exp(value)
            }
            return exp(ptr[stride[1]*t_idx + no_speech]) / expsum
        }
    }
    
    func getLastToken(logits: MLShapedArray<Float>) -> Int {
        logits.withUnsafeShapedBufferPointer { ptr, shape, stride in
            let last_token = shape[1] - 1
            guard let max_logit = (0..<shape[2]).map({ ptr[stride[1]*last_token + $0] })
                .enumerated().max(by: { $0.element < $1.element }) else {
                return -1
            }
            return max_logit.offset
        }
    }
    
    func decodeTokens(tokens: [Int]) -> String {
        let tokenstr = tokens.filter({ $0 < decoder.count }).map({ decoder[$0] }).joined()
        let values = Data(tokenstr.map({ UInt8(codemap[$0] ?? 0) }))
        return String(data: values, encoding: .utf8) ?? ""
    }
}
