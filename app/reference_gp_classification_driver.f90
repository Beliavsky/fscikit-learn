program reference_gp_classification_driver
  use sklearn, only: dp,gaussian_process_classifier
  implicit none
  type(gaussian_process_classifier) :: model
  real(dp), allocatable :: x(:,:),train(:,:),query(:,:)
  integer, allocatable :: binary(:),multi(:),target(:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),binary(i),multi(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,8
    model=gaussian_process_classifier(); train=x; target=binary
    select case(j)
    case(2); target=multi
    case(3); model%amplitude=2.5_dp; model%length_scale=.4_dp
    case(4); model%amplitude=.3_dp; model%length_scale=3
    case(5); model%max_iter_predict=1; target=multi
    case(6); model%max_iter_predict=2
    case(7); train(:,p)=train(:,1); target=multi
    case(8); model%amplitude=10; model%length_scale=1.7_dp; target=multi
    end select
    call model%fit(train,target); query=train+.137_dp
    write(prefix,'(a,i0)') 'case',j
    call vector('_classes',real(model%classes_,dp)); call matrix('_pi',model%pi_)
    call matrix('_w',model%w_sqrt_); call matrix('_mode',model%posterior_mode_); call matrix('_alpha',model%alpha_)
    call vector('_l',reshape(model%l_,[size(model%l_)]))
    call vector('_lmls',model%log_marginal_likelihoods_); call vector('_lml',[model%log_marginal_likelihood()])
    call matrix('_train',model%predict_proba(train)); call matrix('_query',model%predict_proba(query))
    call vector('_train_predict',real(model%predict(train),dp)); call vector('_predict',real(model%predict(query),dp))
    call vector('_score',[model%score(train,target)])
  end do
  close(output)
contains
  subroutine vector(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') trim(prefix)//suffix,values
  end subroutine
  subroutine matrix(suffix,values)
    character(*), intent(in) :: suffix
    real(dp), intent(in) :: values(:,:)
    call vector(suffix,reshape(values,[size(values)]))
  end subroutine
end program
