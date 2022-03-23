function print_exit(m) {
    print("Error") > "/dev/stderr";
    print(m) > "/dev/stderr";
    exit 1;
}

BEGIN {
    if (SOS == "") print_exit("SOS must be specified");
    if (EOS == "") print_exit("EOS must be specified");
}

{
    if (NF >= 2) {
        printf("%s ", SOS);
        for (i = 2; i <= NF; i++) {
            printf("%s ", $i);
        }
        print(EOS)
    } else {
        print("");
    }
}
