function print_exit(m) {
    print("Error", m) > "/dev/stderr";
    exit 1;
}

BEGIN {
    if (SOS == "") print_exit("SOS must be specified");
    if (EOS == "") print_exit("EOS must be specified");
    if (N == "") print_exit("N must be specified");

    bufsize = 0;
    idx = 0;
}

{
    if (NF == 0) {
        bufsize = 0;
        idx = 0;
        print("");
    } else {
        buf[idx] = $0;
        idx = (idx + 1 + N) % N;
        bufsize += 1;

        if (bufsize == N) {
            print(SOS, buf[(idx + N - 1) % N], EOS)
            bufsize -= 1;
        } else {
            print("");
        }
    }
}

