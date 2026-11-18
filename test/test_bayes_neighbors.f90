program test_bayes_neighbors
  use sklearn_common
  use sklearn_naive_bayes
  use sklearn_neighbors
  implicit none
  type(gaussian_nb) :: gnb
  type(multinomial_nb) :: mnb
  type(bernoulli_nb) :: bnb
  type(complement_nb) :: cnb
  type(nearest_neighbors) :: nn
  type(kneighbors_classifier) :: knc
  type(kneighbors_regressor) :: knr
  real(dp) :: x(6,2),counts(6,2),target(6)
  real(dp), allocatable :: prob(:,:),d(:,:),graph(:,:)
  integer, allocatable :: idx(:,:)
  integer :: y(6)
  x(:,1)=[-3._dp,-2._dp,-1._dp,1._dp,2._dp,3._dp]; x(:,2)=x(:,1)*2
  y=[-5,-5,-5,7,7,7]
  call gnb%fit(x,y); prob=gnb%predict_proba(x)
  call require(gnb%score(x,y)==1,'Gaussian NB predictions')
  call require(maxval(abs(sum(prob,dim=2)-1))<1.e-12_dp,'Gaussian NB probabilities')
  counts(:,1)=[5._dp,4._dp,6._dp,0._dp,1._dp,0._dp]
  counts(:,2)=[0._dp,1._dp,0._dp,4._dp,5._dp,6._dp]
  call mnb%fit(counts,y); call require(mnb%score(counts,y)==1,'multinomial NB')
  call bnb%fit(counts,y); call require(bnb%score(counts,y)>0.6_dp,'Bernoulli NB')
  call cnb%fit(counts,y); call require(cnb%score(counts,y)==1,'complement NB')
  nn%n_neighbors=2; call nn%fit(x); call nn%kneighbors(x,d,idx)
  call require(all(d(:,1)==0),'nearest neighbor identity')
  graph=nn%kneighbors_graph(x); call require(all(sum(graph,dim=2)==2),'neighbor graph')
  knc%n_neighbors=3; call knc%fit(x,y)
  call require(knc%score(x,y)==1,'KNN classification')
  knr%n_neighbors=2; knr%weights='distance'; target=x(:,1)**2; call knr%fit(x,target)
  call require(maxval(abs(knr%predict(x)-target))<1.e-12_dp,'KNN exact-match weighting')
  ! Refitting must resize all fitted arrays cleanly.
  call gnb%fit(x(:4,:),y(:4)); call bnb%fit(counts(:4,:),y(:4))
  print *, 'naive Bayes and neighbors: PASS'
end program
