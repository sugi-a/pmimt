function print_exit(m) {
    print("Error") > "/dev/stderr";
    print(m) > "/dev/stderr";
    exit 1;
}

BEGIN {
    if (MAX == "") print_exit("MAX must be specified");
}

{
    if (NF > MAX) print("");
    else print($0);
}
