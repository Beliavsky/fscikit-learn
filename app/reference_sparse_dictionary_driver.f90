program reference_sparse_dictionary_driver
  use sklearn, only: dp,sparse_pca,dictionary_learning,sparse_coder,sparse_encode
  implicit none
  type(sparse_pca) :: projection
  type(dictionary_learning) :: learner
  type(sparse_coder) :: coder
  real(dp), allocatable :: x(:,:),atoms(:,:),warm(:,:),code(:,:),z(:,:),train(:,:)
  integer :: n,p,seed,i,j,c,input,output,width,limit
  logical :: positive
  character(12) :: algorithm
  character(512) :: path
  character(24) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) n,p,seed; allocate(x(n,p),atoms(4,p),warm(n,4))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input)
  do i=1,4
    do j=1,p
      atoms(i,j)=sin(.37_dp*i*j)+.1_dp*cos(real(i+j,dp))
    end do
    atoms(i,:)=atoms(i,:)/sqrt(sum(atoms(i,:)**2))
  end do
  do i=1,n
    do j=1,4
      warm(i,j)=.1_dp*sin(.21_dp*i*j)
    end do
  end do
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  do c=1,34
    write(prefix,'(a,i0)') 'case',c
    if(c<=12) then
      projection=sparse_pca(); projection%n_components=3; projection%alpha=.3_dp
      projection%max_iter=20; projection%random_state=seed
      if(c==2.or.c==5.or.c==8.or.c==11) projection%method='cd'
      select case(c)
      case(3); projection%n_components=0; projection%alpha=.1_dp; projection%max_iter=12
      case(4); projection%n_components=p+2; projection%alpha=.2_dp; projection%max_iter=12
      case(5); projection%alpha=.8_dp; projection%max_iter=35; projection%tol=1.e-4_dp; projection%ridge_alpha=.2_dp
      case(6); projection%alpha=1.e5_dp; projection%max_iter=8
      case(7); projection%max_iter=0
      case(8); projection%n_components=2; projection%alpha=.2_dp; projection%max_iter=12; projection%ridge_alpha=0
      case(9); projection%alpha=.4_dp; projection%tol=.1_dp
      case(10,11); projection%u_init=x(:,:3); projection%v_init=atoms(:3,:); projection%max_iter=15
      case(12); projection%alpha=0; projection%max_iter=3; projection%ridge_alpha=0
      end select
      z=projection%fit_transform(x)
      call matrix('_components',projection%components_); call vector('_mean',projection%mean_)
      call vector('_errors',projection%error_)
      call vector('_shape',real([projection%n_features_in_,projection%n_components_,projection%n_iter_],dp))
      call matrix('_train',z); call matrix('_query',projection%transform(x+.137_dp))
      call matrix('_inverse',projection%inverse_transform(z))
    else if(c<=24) then
      learner=dictionary_learning(); learner%n_components=3; learner%alpha=.3_dp
      learner%max_iter=20; learner%random_state=seed; train=x
      if(c==14.or.c==16.or.c==17.or.c==20.or.c==21.or.c==23) learner%fit_algorithm='cd'
      select case(c)
      case(14); learner%transform_algorithm='lasso_cd'
      case(15)
        learner%n_components=p+2; learner%alpha=.2_dp; learner%max_iter=12
        learner%transform_algorithm='threshold'; learner%transform_alpha=.1_dp; learner%split_sign=.true.
      case(16); learner%alpha=.2_dp; learner%positive_dict=.true.; learner%transform_algorithm='lasso_lars'
      case(17)
        learner%alpha=.2_dp; learner%positive_dict=.true.; learner%positive_code=.true.
        learner%transform_algorithm='lasso_cd'; train=abs(x)
      case(18); learner%alpha=1.e5_dp; learner%max_iter=8; learner%transform_algorithm='lasso_lars'
      case(19); learner%max_iter=0; learner%transform_algorithm='lars'; learner%transform_n_nonzero_coefs=2
      case(20); learner%max_iter=15; learner%transform_n_nonzero_coefs=2
      case(21,22)
        learner%max_iter=15; learner%dict_init=atoms(:3,:); learner%code_init=warm(:,:3)
        learner%transform_algorithm='lasso_lars'; learner%transform_alpha=.15_dp
      case(23)
        learner%max_iter=15; learner%transform_algorithm='lars'; learner%transform_n_nonzero_coefs=2
        learner%split_sign=.true.
      case(24); learner%tol=.1_dp; learner%transform_algorithm='threshold'
      end select
      code=learner%fit_transform(train)
      call matrix('_components',learner%components_); call vector('_errors',learner%error_)
      call vector('_shape',real([learner%n_features_in_,learner%n_iter_],dp)); call matrix('_fit_code',code)
      call matrix('_train',learner%transform(train)); call matrix('_query',learner%transform(train+.137_dp))
    else
      algorithm='lasso_lars'
      select case(c)
      case(26,28); algorithm='lasso_cd'
      case(29); algorithm='lars'
      case(30,33); algorithm='omp'
      case(31,32); algorithm='threshold'
      end select
      width=0; if(c==29.or.c==30) width=2
      positive=c==27.or.c==28.or.c==32; limit=1000; if(c==34) limit=0
      if(c==26) then
        code=sparse_encode(x,atoms,algorithm=trim(algorithm),alpha=.2_dp,n_nonzero_coefs=width, &
          positive=positive,max_iter=limit,init=warm)
      else
        code=sparse_encode(x,atoms,algorithm=trim(algorithm),alpha=.2_dp,n_nonzero_coefs=width, &
          positive=positive,max_iter=limit)
      end if
      coder=sparse_coder(); coder%dictionary=atoms; coder%transform_algorithm=algorithm; coder%transform_alpha=.2_dp
      coder%transform_n_nonzero_coefs=width; coder%positive_code=positive; coder%transform_max_iter=limit
      coder%split_sign=c==31; call coder%fit(x(:0,:)); z=coder%fit_transform(x)
      call matrix('_direct',code); call matrix('_coder',z)
      call vector('_shape',real([size(coder%dictionary,1),size(coder%dictionary,2),size(z,2)],dp))
    end if
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
