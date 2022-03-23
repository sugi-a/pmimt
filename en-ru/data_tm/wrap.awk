BEGIN{
    SOS = "<s> "
    EOS = " </s>"
}
$0{
    print(SOS $0 EOS)
}
