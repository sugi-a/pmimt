BEGIN{
    if (MAX=="") {
        print("MAX must be specified") > "/dev/stderr";
        exit 1;
    }
    FS = "\t";
}

{
    a = split($1, _, " ");
    b = split($2, _, " ");

    if (a <= MAX && b <= MAX) {
        print($1 "\t" $2);
    } else {
        print("");
    }
}
