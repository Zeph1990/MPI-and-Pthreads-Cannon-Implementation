/*
PROGETTO 1
ALGORTIMO DI CANNON SFRUTTANDO UNA FUNZIONE MATRICE PER MATRICE MULTITHREAD
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "mpi.h"
#include "c_timer.h"



void *thread(void *);
void randomMat(int ,int , float (*)[]);
void stampaMat (int ,int , float (*)[] );
void cannon(MPI_Comm, int , int, int, int , float (*)[],float (*)[],float (*)[], int );
void Matikj(int , int, int, int ,float (*)[], float (*)[], float (*)[] );
void matmatthread (int , int, int, int , int , int , int , float (*)[], float (*)[], float (*)[]);
int mod(int , int); 

/* 
Questa struttura dati contiene tutti i dati che devono essere trasmessi a ciascun thread
*/

typedef struct  {
 int nrow;
 int ncol;
 int idThread;
 int LDA;
 int LDB;
 int LDC;
 int nt;
 int dim;
 float *A;
 float *B;
 float *C;
 
}infostruct;

int main(int argc, char* argv[]) {
if(argc<2) {
	printf("ERRORE: e' necessario specificare due parametri di input. Esempio ./prog dimMatrice NThreads.\n");
	return 0; 
}
//INIZIALIZZAZIONE AMBIENTE MPI
int rank,nproc;
MPI_Init(&argc, &argv);

//DA QUI IN POI L'ALGORITMO E' SVOLTO IN PARALLELO
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&nproc);
double tiloc,tfloc,tfglob;

//LEADING DIMENSION DELLE TRE MATRICI
int LDA=500000;
int LDB=500000;
int LDC=500000;
float *A;
float *B;
float *C;
int N=atoi(argv[1]); //DIMENSIONI DELLE MATRICI LOCALI(NxN).
int nt=atoi(argv[2]); //Numero di thread
A=(float*)malloc(sizeof(float)*N*LDA);
B=(float*)malloc(sizeof(float)*N*LDB);
C=(float*)calloc(N*LDC,sizeof(float));

//INIZIALIZZO LE MATRICI LOCALI A E B
srand(rank);
randomMat(LDA,N,(float (*)[])A);
randomMat(LDB,N,(float (*)[])B);

//ESEGUO CANNON
tiloc=get_cur_time();
cannon(MPI_COMM_WORLD,LDA, LDB, LDC, N, (float (*)[])A, (float (*)[])B, (float (*)[])C, nt);
tfloc=get_cur_time()-tiloc;

//CALCOLO DEL TEMPO FINALE D'ESECUZIONE
MPI_Reduce(&tfloc, &tfglob, 1, MPI_DOUBLE, MPI_MAX, 0,MPI_COMM_WORLD);
if(rank==0)
	printf("Tempo finale di esecuzione: %f secondi. \n", tfglob);

MPI_Finalize();
return 0;
}


void cannon (MPI_Comm comm, int LDA, int LDB, int LDC, int N, float A[][LDA],float B[][LDB],float C[][LDC], int nt) {

	//CREO DEI BUFFER
	float *Abuffero;
	float *Bbuffero;
	float *Abufferi;
	float *Bbufferi;

	int rank,nproc;
	MPI_Status status;
	MPI_Request request;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
	Abuffero=(float*)malloc(sizeof(float)*N*LDA);
	Bbuffero=(float*)malloc(sizeof(float)*N*LDB);
	Abufferi=(float*)malloc(sizeof(float)*N*LDA);
	Bbufferi=(float*)malloc(sizeof(float)*N*LDA);

	//COPIA DELLE MATRICI NEI BUFFER
	int i=0;
	int j=0;
	for(i=0; i<N; i++) {
    		for(j=0; j<N; j++) {
			Abuffero[i*N+j]= A[i][j];
			Bbuffero[i*N+j]= B[i][j];
		}
	}

	//CALCOLO DIMENSIONI DELLE SOTTOMATRICI LOCALI PER CIASCUN THREAD
	int nrow=sqrt(nt);
	int ncol=sqrt(nt);
   	int size= sqrt(nproc); //CALCOLO LA DIMENSIONE DELLA GRIGLIA DEI PROCESSORI. NPROC E' IL NUMERO DI PROCESSORI.
   	int myrow = rank / size ;
   	int mycol = rank % size ;

	//INCLINAZIONE INIZIALE  DI A. GLI ELEMENTI POSTI SULLA I-ESIMA RIGA SHIFTANO DI I POSIZIONI A SINISTRA
	
	//CALCOLO IL RANK DEL PROCESSORE DESTINATARIO DEI DATI
	int dest = rank - myrow;
	if (dest < myrow*size)
		dest=dest + size;

	//CALCOLO IL RANK DEL PROCESSORE CHE INVIERA' I DATI A QUESTO PROCESSORE
	int mitt = rank + myrow;
	if (mitt >= (myrow+1)*size)
		mitt=mitt-size;
	
	//INVIO E RICEVO I DATI
	MPI_Isend(Abuffero,N*N,MPI_FLOAT,dest,dest,comm, &request);
	MPI_Irecv(Abufferi,N*N,MPI_FLOAT,mitt,rank,comm, &request);
	MPI_Wait(&request,&status);
	
	//INCLINAZIONE INIZIALE  DI B. GLI ELEMENTI POSTI SULLA I-ESIMA COLONNA SHIFTANO DI I POSIZIONI IN ALTO
	
	//CALCOLO IL RANK DEL PROCESSORE DESTINATARIO DEI DATI
	dest = mod(rank-size*mycol,nproc);
	
	//CALCOLO IL RANK DEL PROCESSORE DESTINATARIO DEI DATI
	mitt = mod(rank+size*mycol,nproc);
	
	//INVIO E RICEVO I DATI
	MPI_Isend(Bbuffero,N*N,MPI_FLOAT,dest,dest+80,comm, &request);
	MPI_Irecv(Bbufferi,N*N,MPI_FLOAT,mitt,rank+80,comm, &request);
	MPI_Wait(&request,&status);
	
	//INIZIO DEL CICLO DI CANNON
	i=0;
	for(i=0; i<size; i++) {
		
		//LA FUNZIONE MATMATTHREAD EFFETTUA IL CALCOLO MATRICE*MATRICE SFRUTTANDO PIU' THREAD.
		matmatthread(N, N ,N , LDC, nt, nrow, ncol, (float (*)[])Abufferi, (float (*)[])Bbufferi, (float (*)[])C);
		
		//COPIO IL BUFFER DI INPUT NEL BUFFER DI OUTPUT. IN QUESTO MODO SONO PRONTO A INVIARE I PROSSIMI DATI
		Abuffero=&Abufferi[0];
		Bbuffero=&Bbufferi[0];

		//ALL'ULTIMO STEP E' INUTILE EFFETTUARE GLI SHIFT
		if(i < size -1 ) {
		
		//SHIFT DI TUTTI GLI ELEMENTI DI A DI UNO A SINISTRA.
		//CALCOLO, COME SEMPRE, DESTINATARIO E MITTENTE
		if(mycol == 0 ) 
		     dest=rank+(size-1);
		else		
		     dest=rank - 1;
		if(mycol == (size-1))
		     mitt = (rank-size)+1;
		else
	             mitt= rank + 1;

		//INVIO E RICEVO I DATI
		MPI_Isend(Abuffero,N*N,MPI_FLOAT,dest,dest+160,comm, &request);
		MPI_Irecv(Abufferi,N*N,MPI_FLOAT,mitt,rank+160,comm, &request);
		MPI_Wait(&request,&status);

		//SHIFT DI TUTTI GLI ELEMENTI DI B DI UNO SOPRA.
		//CALCOLO, COME SEMPRE, DESTINATARIO E MITTENTE
		if(myrow == 0 ) 
		     dest=rank+(size*(size-1));
		else		
		     dest=rank - size;
		if(myrow == (size-1))
		     mitt = rank-(size*(size-1));
		else
	             mitt= rank + size;

		//INVIO E RICEVO I DATI
		MPI_Isend(Bbuffero,N*N,MPI_FLOAT,dest,dest+240,comm, &request);
		MPI_Irecv(Bbufferi,N*N,MPI_FLOAT,mitt,rank+240,comm, &request);
		MPI_Wait(&request,&status);
		
	}
 }
}

//CREA UNA MATRICE FLOAT RANDOM
void randomMat(int LD,int N, float a[][LD]) {
	int i=0;
	int j=0;
	for(i=0; i<N; i++) {
    		for(j=0; j<N; j++) {
			a[i][j]=(100.0*rand()/RAND_MAX);
		}
	}
}


//EFFETTUA UN CALCOLO MATRICE*MATRICE SFRUTTANDO PIU' THREAD.
void matmatthread(int N, int LDA, int LDB, int LDC, int nt, int nrow, int ncol, float A[][LDA], float B[][LDB], float C[][LDC]) {

	//ALLOCO LE STRUTTURE NECESSARIE ALLA CREAZIONE DEI THREAD
	infostruct *info;
	pthread_t *thread_id;
	thread_id = (pthread_t *)calloc(nt,sizeof(pthread_t));

	//ALLOCO E POPOLO LA STRUTTURA DATI INFO PER CIASCUN THREAD
	int i=0;
	for(i=0; i < nt; i++){
    		info = (infostruct *) malloc (sizeof(infostruct));
    		info->idThread = i;
    		info->nrow= nrow;
    		info->ncol = ncol;
    		info->dim = N;
    		info->LDA=LDA;
    		info->LDB=LDB;
    		info->LDC=LDC;
    		info->nt=nt;
    		info->A=*A;
    		info->B=*B;
    		info->C=*C;

    		//LANCIO IL THREAD
    		pthread_create(&thread_id[i], NULL, &thread, (void*)info);
	}

	//ATTENDO CHE I THREAD CONCLUDANO
	i=0;
    	for(i=0; i < nt; i++){
    		pthread_join(thread_id[i], NULL);
   	}

}

//LA FUNZIONE CHE ESEGUE CIASCUN THREAD
void *thread (void *arg){    
	
	//ALLOCO LA STRUTTURA DATI PER RECUPERARE TUTTI I PARAMETRI E LI INSERISCO ALL'INTERNO DI VARIABILI
  	infostruct *info; 
	info=(infostruct *)malloc (sizeof(infostruct));
	info = (infostruct *) arg;
	float *A;
	float *B;
	float *C;
	int LDA;
	int LDB;
	int LDC;
	int Nb;
	int id;
	int nt;
	int N = info->dim;
	nt=info->nt;
	A=info->A;
	B=info->B;
	C=info->C;
	LDA=info->LDA;
	LDB=info->LDB;
	LDC=info->LDC;
	Nb=info->nrow;
	id=info->idThread;
	int irow=id/Nb;
	int icol=id%Nb;

	//ESEGUO RADICE QUADRATA DEL NUMERO DI THREAD VOLTE L'ALGORITMO BASE DI MATRICE PER MATRICE
	int j=0;
	for(j=0; j<sqrt(nt); j++) {

		//LANCIO L'ALGORTIMO MATRICE PER MATRICE PASSANDO CORRETTAMENTE I PUNTI DI PARTENZA DI CIASCUNA MATRICE.  I PUNTI VENGONO TRASMESSI ATTRAVERSO LO SPOSTAMENTO.(OFFSET)
		Matikj(LDA,LDB,LDC,N/Nb,(float (*)[])(A+(N/Nb)*(irow*LDA+j)),(float (*)[])(B+(N/Nb)*(LDB*j+icol)),(float (*)[])(C+(N/Nb)*(irow*LDC+icol)));
		
	}
}

//ALGORTIMO BASE MATRICE PER MATRICE CON LA COMBINAZIONE DI INDICI PIU' EFFICIENTE
void Matikj(int LDA, int LDB, int LDC, int N,float A[][LDA], float B[][LDB], float C[][LDC] ){
	int i=0;
	int j=0;
	int k=0;
	for(i=0;i<N;i++){
		for(k=0;k<N;k++){
			for(j=0;j<N;j++){
				C[i][j]+=A[i][k]*B[k][j];
			}
		}
	}
}

//FUNZIONE AUSILIARIA MODULO. CALCOLA IL MODULO ANCHE DEI NUMERI NEGATIVI CHE C NON CALCOLA NATIVAMENTE.
int mod(int a, int m) {
	if(a>=0) 
	   return a%m;
	else {	
		while (a<0) {
			a+=m;
		}
	}
return a;

}

//FUNZIONE PER LA STAMPA DELLA MATRICE
void stampaMat(int LD, int N, float a[][LD]){
	int i=0;
	int j=0;
	for(i=0; i<N; i++) {
		for(j=0; j<N; j++) {
			printf("%f ",a[i][j]);
		}
		printf("\n");
	}	
}
