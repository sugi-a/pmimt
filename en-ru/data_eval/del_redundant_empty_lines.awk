BEGIN {
    last_is_empty = 1;
}

{
    if (NF == 0) {
        if (!last_is_empty) {
            last_is_empty = 1;
            print("")
        }
    } else {
        last_is_empty = 0;
        print($0)
    }
}
