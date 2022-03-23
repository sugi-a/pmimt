{
    if (NF > 1) {
        for (i = 2; i < NF; i++)
            printf("%s ", $i);
        print($NF)
    } else {
        print("")
    }
}
