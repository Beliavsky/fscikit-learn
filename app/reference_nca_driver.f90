program reference_nca_driver
  use sklearn_nca
  use sklearn_common, only: dp
  implicit none
  type(neighborhood_components_analysis) :: model
  real(dp), allocatable :: x(:,:),q(:,:),a(:,:),gradient(:,:),z(:,:),initial(:,:),history(:,:),search(:,:)
  real(dp) :: objective,tol
  integer, allocatable :: y(:)
  integer :: n,p,k,m,seed,maxiter,kind,matrix,warm,callback_count
  character(16) :: method
  character(60), allocatable :: statuses(:)
  read(*,*) kind,n,p,k,m,seed,maxiter,tol,matrix,warm
  read(*,*) method
  allocate(x(n,p),q(m,p),y(n),a(merge(p,k,k==0),p))
  read(*,*) x;read(*,*) y;read(*,*) q;read(*,*) a
  if(kind==1) then
    allocate(gradient(size(a,1),p))
    call nca_objective_gradient(x,y,a,objective,gradient)
    print '(*(es25.16e3,1x))',objective
    print '(*(es25.16e3,1x))',gradient
    stop
  end if
  model%n_components=k;model%init=method;model%random_state=seed;model%tol=tol;model%max_iter=maxiter
  if(matrix==1) model%init_matrix=a
  allocate(history(maxiter,size(a)),search(maxiter,4),statuses(maxiter));history=0;search=0;statuses=''
  callback_count=0;model%callback=>on_iteration
  if(warm>0) model%max_iter=warm
  call model%fit(x,y);initial=model%initial_components_
  if(warm>0) then
    model%warm_start=.true.;model%max_iter=maxiter;callback_count=0
    call model%fit(x,y)
  end if
  print *,size(model%components_,1),model%n_features_in_,model%n_iter_,model%n_optimizer_iter_, &
    model%n_function_evaluations_,merge(1,0,model%converged_),callback_count
  print '(*(es25.16e3,1x))',initial
  print '(*(es25.16e3,1x))',model%components_
  z=model%transform(x);print '(*(es25.16e3,1x))',z
  z=model%transform(q);print '(*(es25.16e3,1x))',z
  print '(*(es25.16e3,1x))',model%objective_
  print '(*(es25.16e3,1x))',model%gradient_
  print '(*(es25.16e3,1x))',model%initial_components_
  print '(*(es25.16e3,1x))',history(:callback_count,:size(model%components_))
  print '(a)',trim(model%optimization_message_)
  print '(*(es25.16e3,1x))',search(:callback_count,:)
  do matrix=1,callback_count
    print '(a)',trim(statuses(matrix))
  end do
contains
  subroutine on_iteration(transformation,iteration)
    real(dp), intent(in) :: transformation(:)
    integer, intent(in) :: iteration
    if(iteration/=callback_count+1) error stop 'NCA callback sequence'
    if(allocated(model%components_)) then
      if(size(transformation)/=size(model%components_)) error stop 'NCA callback shape'
    end if
    callback_count=callback_count+1
    history(callback_count,:size(transformation))=transformation
    search(callback_count,:)=[model%line_search_bracket_,model%line_search_step_, &
      merge(1._dp,0._dp,model%line_search_bracketed_)]
    statuses(callback_count)=model%line_search_status_
  end subroutine
end program
