BEGIN {
    if (SUFFIX == "") {
        print("Error") > "/dev/stderr";
        exit 1;
    }
}

{
    if (NR <= 5000) print($0) > "./test-1" SUFFIX;
    else if (NR <= 10000) print($0) > "./test-2" SUFFIX;
    else print($0) > "./test-3" SUFFIX;
}
