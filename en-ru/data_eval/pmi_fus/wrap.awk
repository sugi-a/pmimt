{
    if (NF > 0) {
        print("<s> " $0 " </s>");
    } else {
        print("")
    }
}
