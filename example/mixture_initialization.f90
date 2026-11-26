program mixture_initialization
 use sklearn,only:dp,gaussian_mixture
 implicit none
 type(gaussian_mixture)::model
 real(dp)::x(40,2)
 real(dp),allocatable::draws(:,:)
 integer,allocatable::labels(:),sample_labels(:)
 integer::i
 do i=1,40
  x(i,:)=[sin(real(i,dp)),cos(2._dp*i)]+merge(0._dp,6._dp,i<=20)
 end do
 model%n_components=2;model%init_params='random_from_data';model%n_init=3
 model%random_state=42;model%max_iter=5;model%warm_start=.true.
 labels=model%fit_predict(x)
 model%max_iter=50;call model%fit(x)
 call model%sample(8,draws,sample_labels)
 print *,'EM iterations:',model%n_iter_,'converged:',model%converged_
 print *,'Mean log likelihood:',model%score(x)
 print *,'Sample component labels:',sample_labels
 do i=1,size(draws,1)
  print *,draws(i,:)
 end do
end program
