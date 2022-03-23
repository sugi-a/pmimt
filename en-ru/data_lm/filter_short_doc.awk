function print_exit(m) {
    print("Error") > "/dev/stderr";
    print(m) > "/dev/stderr";
    exit 1;
}

BEGIN {
    if (MIN_TOKS == "") print_exit("MIN_TOKS not specified.");

    print("MIN_TOKS:", MIN_TOKS) > "/dev/stderr";

    count = 0;
    bufsize = 0;
}

{
    if ($0 == "") {
        count = 0;
        bufsize = 0;

        print("");
    } else {
        count += NF;

        buf[bufsize] = $0;
        bufsize += 1;

        if (count >= MIN_TOKS) {
            for (i = 0; i < bufsize; i++) {
                print(buf[i]);
            }
            bufsize = 0;
        }
    }
}
