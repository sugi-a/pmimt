type Int = number;

export type Config = {
    "size": Int,
    "PAD_ID": Int,
    "SOS_ID": Int,
    "EOS_ID": Int,
    "UNK_ID": Int,
    "source_dict": string,
    "target_dict": string,
};