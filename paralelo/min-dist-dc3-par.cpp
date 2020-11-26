/* min-dist-dc3.cpp (Roland Teodorowitsch; 29 out. 2020)
 * Compilation: mpic++ -o min-dist-dc-mpi min-dist-dc3.cpp -lm
 * Note: Includes some code from the sequential solution of the
 *       "Closest Pair of Points" problem from the
 *       14th Marathon of Parallel Programming avaiable at
 *       http://lspd.mackenzie.br/marathon/19/points.zip
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <mpi.h>

#define SIZE 10000000
#define START 1000000
#define STEP  1000000

#define EPS 0.00000000001
#define BRUTEFORCESSIZE 200

using namespace std;

typedef struct {
    double x;
    double y;
} point_t;

point_t points[SIZE];
point_t border[SIZE];

unsigned long long llrand() {
    unsigned long long r = 0;
    int i;
    for (i = 0; i < 5; ++i)
        r = (r << 15) | (rand() & 0x7FFF);
    return r & 0xFFFFFFFFFFFFFFFFULL;
}

void points_generate(point_t *points, int size, int seed) {
    int p, i, found;
    double x, y;
    srand(seed);
    p = 0;
    while (p<size) {
        x = ((double)(llrand() % 20000000000) - 10000000000) / 1000.0;
        y = ((double)(llrand() % 20000000000) - 10000000000) / 1000.0;
        if (x >= -10000000.0 && x <= 10000000.0 && y >= -10000000.0 && y <= 10000000.0) {
            points[p].x = x;
            points[p].y = y;
            p++;
        }
    }
}

bool compX(const point_t &a, const point_t &b) {
    if (a.x == b.x)
        return a.y < b.y;
    return a.x < b.x;
}

bool compY(const point_t &a, const point_t &b) {
    if (a.y == b.y)
        return a.x < b.x;
    return a.y < b.y;
}

double points_distance_sqr(point_t *p1, point_t *p2) {
    double dx, dy;
    dx = p1->x - p2->x;
    dy = p1->y - p2->y;
    return dx*dx + dy*dy;
}

double points_min_distance_dc(point_t *point,point_t *border,int l, int r, int p, int id) {
    double minDist = DBL_MAX;
    double dist, dL, dR;
    int i, j;
    int m, tamanho, tamanho1, tamanho2;
    int pai, filho1, filho2;
    MPI_Status status;

    if (r-l+1 <= BRUTEFORCESSIZE) {
        for (i=l; i<r; i++){
            for (j = i+1; j<=r; j++) {
                dist = points_distance_sqr(point+i, point+j);
                if (dist<minDist) {
                    minDist = dist;
                }
            }
        }
        return minDist;
    }

    if (id >= 0) {
        if (id == 0) {

            m = (l+r)/2;

            // DIVIDE 1
            filho1 = id*2+1;
            MPI_Send((void *)&m, 1, MPI_INT, filho1, 1, MPI_COMM_WORLD);
            MPI_Send((void *)&point[l], m*2, MPI_DOUBLE, filho1, 2, MPI_COMM_WORLD);

            // DIVIDE 2
            filho2 = filho1+1;
            tamanho2 = r-m;
            MPI_Send((void *)&tamanho2, 1, MPI_INT, filho2, 1, MPI_COMM_WORLD);
            MPI_Send((void *)&point[m], tamanho2*2, MPI_DOUBLE, filho2, 2, MPI_COMM_WORLD);

            MPI_Recv((void *)&dL, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Recv((void *)&dR, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            minDist = (dL < dR ? dL : dR);
        }
        else {
            MPI_Recv((void *)&tamanho, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            pai = status.MPI_SOURCE;
            MPI_Recv((void *)&point[0], tamanho*2, MPI_DOUBLE, pai, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            l = 0;
            r = tamanho;
            m = (l+r)/2;

            if (id >= p/2) {
                // NODO FOLHA
                dL = points_min_distance_dc(point,border,l,m,-1,-1);
                dR = points_min_distance_dc(point,border,m,r,-1,-1);
            } 
            else {
                // NODO INTERMEDIÁRIO

                // DIVIDE 1
                filho1 = id*2+1;
                tamanho1 = m;
                MPI_Send((void *)&tamanho1, 1, MPI_INT, filho1, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&point[l], tamanho1*2, MPI_DOUBLE, filho1, 2, MPI_COMM_WORLD);

                // DIVIDE 2
                filho2 = filho1+1;
                tamanho2 = r-m;
                MPI_Send((void *)&tamanho2, 1, MPI_INT, filho2, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&point[tamanho1], tamanho2*2, MPI_DOUBLE, filho2, 2, MPI_COMM_WORLD);

                MPI_Recv((void *)&dL, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                MPI_Recv((void *)&dR, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            }
            
            minDist = (dL < dR ? dL : dR);
        }
    }
    else {
        m = (l+r)/2;
        
        dL = points_min_distance_dc(point,border,l,m,-1,-1);
        dR = points_min_distance_dc(point,border,m,r,-1,-1);

        minDist = (dL < dR ? dL : dR);
    }

    int k = l;
    for(i=m-1; i>=l && fabs(point[i].x-point[m].x)<minDist; i--)
        border[k++] = point[i];
    for(i=m+1; i<=r && fabs(point[i].x-point[m].x)<minDist; i++)
        border[k++] = point[i];


    if (k-l <= 1) {
        if (id > 0)
            MPI_Send((void *)&minDist, 1, MPI_DOUBLE, pai, 0, MPI_COMM_WORLD);
        return minDist;
    }

    sort(&border[l], &border[l]+(k-l), compY);

    for (i=l; i<k; i++) {
        for (j=i+1; j<k && border[j].y - border[i].y < minDist; j++) {
            dist = points_distance_sqr(border+i, border+j);
            if (dist < minDist)
                minDist = dist;
        }
    }

    if (id > 0)
        MPI_Send((void *)&minDist, 1, MPI_DOUBLE, pai, 0, MPI_COMM_WORLD);
    return minDist;
}

int main(int argc, char *argv[]) {
    int i, p, id;
    double elapsed_time, output;
   
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (id == 0) {
        points_generate(points,SIZE,11);
        sort(&points[0], &points[SIZE], compX);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (i=START; i<=SIZE; i+=STEP) {
        if (id == 0) {
            elapsed_time = -MPI_Wtime();
            output = sqrt(points_min_distance_dc(points,border,0,i-1,p,id));
            printf("%.6lf\n", output);
            elapsed_time += MPI_Wtime();  
            fprintf(stderr,"%d %lf\n",i,elapsed_time);
        } else {
            points_min_distance_dc(points,border,0,i-1,p,id);
        }
    }
    MPI_Finalize();
    return 0;
}

