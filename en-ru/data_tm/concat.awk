BEGIN{
    CONC = " </s> "
    SEP = " </s> "
    M = 4
    s = 0
    e = 0
}
{
    if(!$0){
        s = 0
        e = 0
    }else{
        if(e - s == M){
            s += 1
        }
        a[e % M] = $0
        e += 1
        for(i = s; i <= e - 2; i++){
            if (i == s){
                printf("%s", a[i % M])
            }else{
                printf("%s", SEP a[i % M])
            }
        }
        printf("%s\n", CONC a[(e-1) % M])
    }
}
