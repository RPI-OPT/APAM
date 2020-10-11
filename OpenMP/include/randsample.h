#include <random>
#include <vector>

double GetUniform()
{
    static std::default_random_engine re;
    static std::uniform_real_distribution<double> Dist(0,1);
    return Dist(re);
}

void SampleWithoutReplacement
(
 int populationSize,    // size of set sampling from
 int sampleSize,        // size of each sample
 std::vector<int> & samples  // output, zero-offset indicies to selected items
)
{
    std::default_random_engine re;
    std::uniform_real_distribution<double> Dist(0,1);
    // Use Knuth's variable names
    int& n = sampleSize;
    int& N = populationSize;
    
    int t = 0; // total input records dealt with
    int m = 0; // number of items selected so far
    double u;
    
    while (m < n)
    {
        // caution: not good for multi-threaded program to call GetUniform() because of the static structure, which will block the threads that call GetUniform later
        
        //u = GetUniform(); // call a uniform(0,1) random number generator
        
        u = Dist(re);
        
        if ( (N - t)*u >= n - m )
        {
            t++;
        }
        else
        {
            samples[m] = t;
            t++; m++;
        }
    }
}
