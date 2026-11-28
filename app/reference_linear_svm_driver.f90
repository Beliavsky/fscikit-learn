program reference_linear_svm_driver
  use sklearn_common,only:dp
  use sklearn_linear_svm,only:linear_svc,linear_svr
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  implicit none
  type(linear_svc) :: classifier
  type(linear_svr) :: reg
  type(csr_matrix)::sx,sq
  real(dp), allocatable :: x(:,:),w(:),target(:),train(:,:),query(:,:),weights(:)
  integer, allocatable :: binary(:),multi(:),labels(:)
  integer :: n,p,i,j,input,output
  character(1024) :: path
  character(32) :: prefix,layout
  logical::sparse
  call get_command_argument(3,layout);sparse=len_trim(layout)>0
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p; allocate(x(n,p),w(n),target(n),binary(n),multi(n))
  do i=1,n
    read(input,*) x(i,:),w(i),target(i),binary(i),multi(i)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do j=1,10
    classifier=linear_svc(); classifier%tol=1.e-12_dp; classifier%max_iter=10000
    train=x; weights=1+0*w; labels=binary
    select case(j)
    case(2); labels=multi
    case(3); classifier%fit_intercept=.false.
    case(4); classifier%c=.05_dp; classifier%intercept_scaling=2
    case(5); labels=multi; weights=w
    case(6); labels=multi; weights=w; classifier%class_weight_balanced=.true.
    case(7); labels=multi; weights=w
    case(8); weights=w
    case(9); train(:,p)=train(:,1); labels=multi
    case(10); classifier%c=10; classifier%intercept_scaling=4
    end select
    if(sparse)then
      sx=sparse_input(train)
      if(j==7)then
        call classifier%fit(sx,labels,sample_weight=weights,class_weight=[.5_dp,2._dp,1.5_dp])
      else if(j==8)then
        call classifier%fit(sx,labels,sample_weight=weights,class_weight=[.5_dp,2._dp])
      else
        call classifier%fit(sx,labels,sample_weight=weights)
      end if
    else if(j==7) then
      call classifier%fit(train,labels,sample_weight=weights,class_weight=[.5_dp,2._dp,1.5_dp])
    else if(j==8) then
      call classifier%fit(train,labels,sample_weight=weights,class_weight=[.5_dp,2._dp])
    else
      call classifier%fit(train,labels,sample_weight=weights)
    end if
    query=train+.137_dp; write(prefix,'(a,i0)') 'svc',j
    call matrix('_coef',classifier%coef_); call vector('_intercept',classifier%intercept_)
    call vector('_classes',real(classifier%classes_,dp))
    if(sparse)then
      sq=sparse_input(query)
      call matrix('_train',classifier%decision_function(sx));call matrix('_query',classifier%decision_function(sq))
      call vector('_predict',real(classifier%predict(sq),dp));call vector('_score',[classifier%score(sx,labels)])
    else
      call matrix('_train',classifier%decision_function(train));call matrix('_query',classifier%decision_function(query))
      call vector('_predict',real(classifier%predict(query),dp));call vector('_score',[classifier%score(train,labels)])
    end if
    call vector('_objective',classifier%objective_)
  end do
  do j=1,8
    reg=linear_svr(); reg%tol=1.e-12_dp; reg%max_iter=10000; train=x; weights=1+0*w
    select case(j)
    case(2); reg%epsilon=.2_dp
    case(3); reg%fit_intercept=.false.; reg%epsilon=.1_dp
    case(4); reg%c=.05_dp; reg%intercept_scaling=2; reg%epsilon=.3_dp
    case(5); weights=w; reg%epsilon=.2_dp
    case(6); reg%c=10; reg%intercept_scaling=4; weights=w; reg%epsilon=.5_dp
    case(7); train(:,p)=train(:,1); reg%epsilon=.15_dp
    case(8); reg%epsilon=100
    end select
    if(sparse)then
      sx=sparse_input(train);call reg%fit(sx,target,sample_weight=weights)
    else
      call reg%fit(train,target,sample_weight=weights)
    end if
    query=train+.137_dp
    write(prefix,'(a,i0)') 'svr',j
    call vector('_coef',reg%coef_); call vector('_intercept',[reg%intercept_])
    if(sparse)then
      sq=sparse_input(query);call vector('_train',reg%predict(sx));call vector('_query',reg%predict(sq))
      call vector('_score',[reg%score(sx,target)])
    else
      call vector('_train',reg%predict(train));call vector('_query',reg%predict(query))
      call vector('_score',[reg%score(train,target)])
    end if
    call vector('_objective',[reg%objective_])
  end do
  close(output)
contains
  function sparse_input(a)result(x)
    real(dp),intent(in)::a(:,:)
    type(csr_matrix)::x
    integer::i,j,k,nz
    if(layout=='csr')then
      x=csr_from_dense(a);return
    end if
    nz=count(a/=0);if(layout=='duplicates')nz=3*nz
    if(layout=='zeros')nz=size(a)
    x%n_rows=size(a,1);x%n_cols=size(a,2)
    allocate(x%indptr(x%n_rows+1),x%indices(nz),x%data(nz));k=1;x%indptr(1)=1
    do i=1,x%n_rows
      do j=x%n_cols,1,-1
        if(layout/='zeros'.and.a(i,j)==0)cycle
        if(layout=='duplicates')then
          x%indices(k:k+2)=j;x%data(k:k+2)=[.25_dp*a(i,j),.75_dp*a(i,j),0._dp];k=k+3
        else
          x%indices(k)=j;x%data(k)=a(i,j);k=k+1
        end if
      end do
      x%indptr(i+1)=k
    end do
  end function
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
