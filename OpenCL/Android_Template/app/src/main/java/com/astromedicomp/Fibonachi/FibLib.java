package com.astromedicomp.Fibonachi;

public class FibLib {
    public FibLib() {
        super();
    }

    public static long fibLibJR(long n)
    {
        return n <= 0? 0: n==1? 1 :fibLibJR(n-1) + fibLibJR(n-2);
    }

    public static long fibJI(long n)
    {
        long previous = -1;
        long result = 1;

        for (long i =0; i<n; i++)
        {
            long sum = result + previous;
            previous=result;
            result =sum;
        }
        return result;
    }

    public static native long fibNR(long n);
    public static native long fibNI(long n);

}
