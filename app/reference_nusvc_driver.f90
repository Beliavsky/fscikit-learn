program reference_nusvc_driver
  use sklearn_nu_classification, only: nu_svc
  use sklearn_common, only: dp
  implicit none
  type(nu_svc) :: model
  real(dp), allocatable :: x(:,:),w(:),query(:,:),z(:,:)
  integer, allocatable :: y(:)
  integer :: n,p,nq,i,input,output,ties
  character(512) :: path
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,nq,ties
  read(input,*) model%kernel,model%gamma_rule,model%decision_function_shape
  read(input,*) model%nu,model%gamma,model%coef0,model%degree,model%tol,model%max_iter
  allocate(x(n,p),y(n),w(n),query(nq,p))
  do i=1,n
    read(input,*) x(i,:),y(i),w(i)
  end do
  do i=1,nq
    read(input,*) query(i,:)
  end do
  close(input); model%break_ties=ties==1
  call get_command_argument(3,path);model%shrinking=trim(path)/='--no-shrinking'
  call model%fit(x,y,w)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  call vector('classes',real(model%classes_,dp)); call vector('support',real(model%support_,dp))
  call vector('n_support',real(model%n_support_,dp)); call matrix('dual',model%dual_coef_)
  call matrix('vectors',model%support_vectors_); call vector('intercept',model%intercept_)
  call matrix('train',model%decision_function(x)); call matrix('query',model%decision_function(query))
  call vector('train_predict',real(model%predict(x),dp))
  call vector('predict',real(model%predict(query),dp)); call vector('score',[model%score(x,y)])
  call matrix('alpha',model%pair_alpha_); call vector('scales',model%pair_scales_)
  call vector('kkt',model%kkt_violation_); call vector('iterations',real(model%n_iter_,dp))
  call vector('meta',[model%gamma_,real(model%n_features_in_,dp),real(model%fit_status_,dp)])
  if(allocated(model%coef_)) call matrix('coef',model%coef_)
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
