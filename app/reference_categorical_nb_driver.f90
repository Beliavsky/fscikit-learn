program reference_categorical_nb_driver
  use sklearn, only: dp,categorical_nb
  implicit none
  type(categorical_nb) :: model
  real(dp), allocatable :: x(:,:),w(:),query(:,:)
  integer, allocatable :: y(:),target(:)
  integer :: n,p,i,j,k,input,output,third
  character(1024) :: path
  character(32) :: prefix,suffix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),w(n),y(n))
  do i=1,n
    read(input,*) x(i,:),w(i),y(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  third=n/3
  do j=1,10
    model=categorical_nb(); target=y
    select case(j)
    case(2); model%alpha=.1_dp
    case(3); model%fit_prior=.false.
    case(5); model%class_prior=[.2_dp,.3_dp,.5_dp]
    case(6); model%min_categories=8
    case(7); model%min_categories_per_feature=[(4+k,k=1,p)]
    case(9); model%class_prior=[1._dp,0._dp,0._dp]
    case(10); target=5
    end select
    if(j==8) then
      call model%partial_fit(x(:third,:),target(:third),classes=[-4,7,21],sample_weight=w(:third))
      call model%partial_fit(x(third+1:2*third,:),target(third+1:2*third),sample_weight=w(third+1:2*third))
      call model%partial_fit(x(2*third+1:,:),target(2*third+1:),sample_weight=w(2*third+1:))
    else if(j==4.or.j==5) then
      call model%fit(x,target,sample_weight=w)
    else
      call model%fit(x,target)
    end if
    query=x
    do k=1,p
      query(:,k)=modulo(x(:,k)+1,real(model%n_categories_(k),dp))
    end do
    write(prefix,'(a,i0)') 'case',j
    call vector('_classes',real(model%classes_,dp)); call vector('_counts',model%class_count_)
    call vector('_prior',model%class_log_prior_); call vector('_categories',real(model%n_categories_,dp))
    call matrix('_joint',model%joint_log_likelihood(query)); call matrix('_logp',model%predict_log_proba(query))
    call matrix('_prob',model%predict_proba(query)); call vector('_predict',real(model%predict(query),dp))
    call vector('_train_predict',real(model%predict(x),dp))
    call vector('_score',[model%score(x,target)])
    do k=1,p
      write(suffix,'(a,i0)') '_category',k; call matrix(trim(suffix),model%category_count_(k)%values)
      write(suffix,'(a,i0)') '_feature',k; call matrix(trim(suffix),model%feature_log_prob_(k)%values)
    end do
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
