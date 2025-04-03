from speechbrain.inference.ASR import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", 
                                           hparams_file='hyperparams_develop.yaml', 
                                           savedir="./pretrained_ASR")

if __name__ == "__main__":
    asr_model.transcribe_file('/scratch/mfron/IST-LAB/JASMIN/kaldi_processed/nl_test_read_all_hires/data/format.7/N000172-fn000418-371.8139-373.6982.wav')