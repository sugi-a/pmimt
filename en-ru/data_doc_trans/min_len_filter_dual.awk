BEGIN{
    FS = "\t"
}

{
    a = split($1, _, " ");
    b = split($2, _, " ");

    if (a > 0 && b > 0) {
        print($1 "\t" $2);
    } else {
        print("");
    }
}
