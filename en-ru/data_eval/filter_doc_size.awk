function print_exit(m) {
    print(m) > "/dev/stderr";
    exit 1;
}

BEGIN {
    if (MIN == "") print_exit("MIN must be specified")

    count = 0;
    ok = 0;
}

{
    if (NF == 0) {
        count = 0;
        ok = 0;
        print("")
    } else {
        if (ok) {
            print($0)
        } else {
            buf[count] = $0;
            count += 1;
            
            if (count >= MIN) {
                ok = 1;
                for (i = 0; i < MIN; i++) {
                    print(buf[i]);
                }
            }
        }
    }
}
