function print_exit(m) {
    print(m) > "/dev/stderr";
    exit 1;
}

BEGIN {
    FS = "\t";

    if (MAX == "") print_exit("MAX must be specified");
}

{
    a = split($1, _, " ")
    b = split($2, _, " ")

    if (a <= MAX && b <= MAX) print($0);
    else print("\t");
}
