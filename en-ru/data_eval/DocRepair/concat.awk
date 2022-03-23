BEGIN {
    BUFSIZE = 3;
    
    filled = 0;
    idx = 0;
}

{
    if (NF > 0) {
        if (filled == BUFSIZE) {
            printf("%s", "<s>");
            for (i = 0; i < BUFSIZE; i++) {
                printf(" %s %s", buf[(idx + i) % BUFSIZE], "</s>")
            }
            printf(" %s %s", $0, "</s>");
            print("");

            buf[idx] = $0;
            idx = (idx + 1) % BUFSIZE;
        } else {
            buf[idx] = $0;
            idx = (idx + 1) % BUFSIZE;
            filled++;
        }
    } else {
        for (i = 0; i < filled; i++) {
            print("");
        }
        filled = 0;
        idx = 0;
        print("");
    }
}

END {
    for (i = 0; i < filled; i++) {
        print("");
    }

}
