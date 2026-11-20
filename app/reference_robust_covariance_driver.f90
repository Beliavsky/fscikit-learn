program reference_robust_covariance_driver
  use sklearn_robust_covariance, only: min_cov_det,elliptic_envelope,fast_mcd
  use sklearn_common, only: dp
  implicit none
  type(min_cov_det) :: model
  type(elliptic_envelope) :: envelope
  real(dp), allocatable :: x(:,:),query(:,:),raw_location(:),raw_covariance(:,:),raw_dist(:),comp(:,:),weights(:)
  logical, allocatable :: raw_support(:)
  integer, allocatable :: labels(:)
  integer :: n,p,nq,center,store,input,output,i
  character(512) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,nq,center,store
  read(input,*) model%support_fraction,model%random_state,envelope%contamination
  allocate(x(n,p),query(nq,p),weights(nq),labels(nq))
  do i=1,n
    read(input,*) x(i,:)
  end do
  do i=1,nq
    read(input,*) query(i,:),labels(i),weights(i)
  end do
  close(input); model%assume_centered=center==1; model%store_precision=store==1
  call fast_mcd(x,raw_location,raw_covariance,raw_support,raw_dist,model%support_fraction,model%random_state)
  call model%fit(x)
  envelope%assume_centered=model%assume_centered; envelope%store_precision=model%store_precision
  envelope%support_fraction=model%support_fraction; envelope%random_state=model%random_state
  call envelope%fit(x)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call vector('fast_location',raw_location); call matrix('fast_covariance',raw_covariance)
  call vector('fast_support',real(merge(1,0,raw_support),dp)); call vector('fast_dist',raw_dist)
  call vector('raw_location',model%raw_location_); call matrix('raw_covariance',model%raw_covariance_)
  call vector('raw_support',real(merge(1,0,model%raw_support_),dp)); call vector('location',model%location_)
  call matrix('covariance',model%covariance_); call matrix('precision',model%get_precision())
  call vector('support',real(merge(1,0,model%support_),dp)); call vector('dist',model%dist_)
  call vector('mahalanobis',model%mahalanobis(query)); call vector('likelihood',[model%score(query)])
  call vector('diagnostics',[model%correction_,model%reweight_threshold_,real(model%rank_,dp), &
    real(model%n_features_in_,dp),real(merge(1,0,allocated(model%precision_)),dp)])
  comp=model%covariance_; comp(1,1)=comp(1,1)+.1_dp
  call vector('norms',[model%error_norm(comp),model%error_norm(comp,norm='spectral'), &
    model%error_norm(comp,scaling=.false.,squared=.false.)])
  call vector('envelope_location',envelope%location_); call matrix('envelope_covariance',envelope%covariance_)
  call vector('samples',envelope%score_samples(query)); call vector('decision',envelope%decision_function(query))
  call vector('predict',real(envelope%predict(query),dp)); call vector('offset',[envelope%offset_])
  call vector('accuracy',[envelope%score(query,labels),envelope%score(query,labels,weights)])
  close(output)
contains
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
end program
