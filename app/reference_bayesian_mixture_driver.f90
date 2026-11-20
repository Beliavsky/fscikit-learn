program reference_bayesian_mixture_driver
  use sklearn_bayesian_mixture, only: bayesian_gaussian_mixture
  use sklearn_common, only: dp
  implicit none
  type(bayesian_gaussian_mixture) :: model
  real(dp), allocatable :: x(:,:),query(:,:),draws(:,:),again(:,:),mean(:),covariance(:,:)
  integer, allocatable :: labels(:),draw_labels(:),again_labels(:)
  integer :: n,p,nq,operation,n_draw,input,output,i,j,c
  logical :: prior_mean,prior_cov,grouped
  character(512) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,nq,operation,n_draw
  read(input,*) model%n_components,model%max_iter,model%n_init,model%random_state,model%tol,model%reg_covar
  read(input,*) model%weight_concentration_prior,model%mean_precision_prior,model%degrees_of_freedom_prior,model%warm_start
  read(input,*) model%covariance_type,model%weight_concentration_prior_type,model%init_params
  read(input,*) prior_mean,prior_cov
  allocate(x(n,p),query(nq,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  do i=1,nq
    read(input,*) query(i,:)
  end do
  if(prior_mean) then
    allocate(model%mean_prior(p)); read(input,*) model%mean_prior
  end if
  if(prior_cov) then
    allocate(model%covariance_prior(p,p))
    do i=1,p
      read(input,*) model%covariance_prior(i,:)
    end do
  end if
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  if(operation==2) then
    labels=model%fit_predict(x); call vector('fit_predict',real(labels,dp))
  else
    call model%fit(x)
  end if
  if(operation>=3) then
    call snapshot('first_',x,query)
    select case(operation)
    case(3); call model%fit(x)
    case(4); call model%fit(x+.2_dp)
    case(5)
      model%warm_start=.false.; model%n_components=2; model%covariance_type='diag'
      if(allocated(model%mean_prior)) deallocate(model%mean_prior)
      if(allocated(model%covariance_prior)) deallocate(model%covariance_prior)
      call model%fit(x(:,:1))
    end select
  end if
  p=model%n_features_in_
  call snapshot('',x(:,:p),query(:,:p))
  if(n_draw>0) then
    call model%sample(n_draw,draws,draw_labels); call model%sample(n_draw,again,again_labels)
    mean=sum(draws,dim=1)/n_draw
    covariance=matmul(transpose(draws-spread(mean,1,n_draw)),draws-spread(mean,1,n_draw))/n_draw
    call vector('sample_mean',mean); call matrix('sample_covariance',covariance)
    call vector('sample_counts',real([(count(draw_labels==c),c=0,model%n_components-1)],dp))
    grouped=all(draw_labels>=0).and.all(draw_labels<model%n_components)
    if(n_draw>1) grouped=grouped.and.all(draw_labels(2:)>=draw_labels(:n_draw-1))
    call vector('sample_meta',real([merge(1,0,grouped),merge(1,0,all(draws==again).and.all(draw_labels==again_labels))],dp))
  end if
  close(output)
contains
  subroutine snapshot(prefix,data,queries)
    character(*), intent(in) :: prefix
    real(dp), intent(in) :: data(:,:),queries(:,:)
    call vector(prefix//'weights',model%weights_); call matrix(prefix//'means',model%means_)
    call matrix(prefix//'concentration',model%weight_concentration_); call vector(prefix//'mean_precision',model%mean_precision_)
    call vector(prefix//'degrees',model%degrees_of_freedom_)
    call cube(prefix//'covariances',model%covariances_); call cube(prefix//'precisions',model%precisions_)
    call cube(prefix//'precision_cholesky',model%precisions_cholesky_)
    call vector(prefix//'lower_bound',[model%lower_bound_]); call vector(prefix//'history',model%lower_bounds_)
    call vector(prefix//'mean_prior',model%mean_prior_); call matrix(prefix//'covariance_prior',model%covariance_prior_)
    call vector(prefix//'scalar_priors',[model%weight_concentration_prior_,model%mean_precision_prior_, &
      model%degrees_of_freedom_prior_])
    call vector(prefix//'predict',real(model%predict(queries),dp)); call matrix(prefix//'proba',model%predict_proba(queries))
    call vector(prefix//'score_samples',model%score_samples(queries)); call vector(prefix//'score',[model%score(data)])
    call vector(prefix//'meta',real([model%n_features_in_,model%n_components,model%n_iter_,merge(1,0,model%converged_)],dp))
  end subroutine
  subroutine cube(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
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
