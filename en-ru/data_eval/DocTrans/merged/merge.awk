BEGIN {
    FS = "\t";
}

{
    if ($1 == "" && $2 != "") print($2);
    else print($1);
}
