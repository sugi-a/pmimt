BEGIN {
    last_is_empty = 1;
}

{
    if ($0 == "") {
        if (!last_is_empty) print("");
        last_is_empty = 1;
    } else {
        last_is_empty = 0;
        print($0)
    }
}
