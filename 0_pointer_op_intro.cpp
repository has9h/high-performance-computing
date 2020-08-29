#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

void printL(int a);
void printL1(int *a);
void printArr(int a[], int siz);
void printL2(int a[], int n);
void printL3(int *a, int n);
void printL4(int *a, int n);

int main()
{
    int a1 = 20;
    cout <<"address:" <<&a1 << endl << "Value:" << a1 <<  endl;
    printL(a1);
    cout << a1 << endl;

    printL1(&a1);
    cout << a1 << endl;

    int A[5] = {0,10,20,30,40};

    cout << "Array A:" << endl;
    printArr(A,5);
    printL2(A,5);
    cout << "After printL2 call:" << endl;;
    printArr(A,5);
    cout  << endl;

    int n = 10;
    int *B =  new int[n];
    srand(time(0));

    for (int i = 0; i < n; i++ )
    {
        B[i] = rand()%100;
    }
     cout << "Array B:" << endl;
     printArr(B,n);
     cout  << endl;

     printL3(B, n);
     cout << "After printL3 call:" << endl;;
     printArr(B,n);
     cout  << endl;

     printL4(B, n);
     cout << "After printL4 call:" << endl;;
     printArr(B,n);
     cout  << endl;


    return 0;
}

void printArr(int a[], int siz)
{
     for (int i = 0; i < siz; i++ )
    {

        cout << a[i] << "  ";
    }
    cout << endl;
}

void printL(int a)
{
    a = 10;
}

void  printL1(int *a)
{
    *a = 3;
}

void  printL2(int a[], int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i*100;
    }
}

void  printL3(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = i*100;
    }
}

void  printL4(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        *(a + i) = i*300;
    }
}