!$0{
    if(c >= M){
        for(i = 1; i <= c; i++){
            print(a[i])
        }
        print("")
    }
    c = 0
}
$0{
    c++;
    a[c] = $0
}
