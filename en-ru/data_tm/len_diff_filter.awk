BEGIN {
    FS = "\t"
    if (SMOOTH == "") exit 1;
    if (RATIO == "") exit 1;
}

{
    a = split($1, _, " ");
    b = split($2, _, " ");
    if (a < b) {
        t = b;
        b = a;
        a = t;
    }

    test = (a + SMOOTH) / (b + SMOOTH) <= RATIO;

    if(test) print($0);
}
