program reference_structured_datasets_driver
  use sklearn_common, only: dp
  use sklearn_datasets, only: make_sparse_coded_signal,make_sparse_spd_matrix
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  real(dp), allocatable :: x(:,:),d(:,:),code(:,:),other(:,:),d2(:,:),c2(:,:)
  type(csr_matrix) :: csr
  integer :: cases,index,kind,n,k,p,s,seed,normalized
  real(dp) :: alpha,low,high,nan,inf
  character(60) :: mode
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    select case(trim(mode))
    case('samples');call make_sparse_coded_signal(0,2,2,1,x,d,code)
    case('components');call make_sparse_coded_signal(2,0,2,1,x,d,code)
    case('features');call make_sparse_coded_signal(2,2,0,1,x,d,code)
    case('support');call make_sparse_coded_signal(2,2,2,0,x,d,code)
    case('oversupport');call make_sparse_coded_signal(2,2,2,3,x,d,code)
    case('coded_seed');call make_sparse_coded_signal(2,2,2,1,x,d,code,-1)
    case('dimension');call make_sparse_spd_matrix(x,n_dim=0)
    case('population');call make_sparse_spd_matrix(x,n_dim=50000)
    case('alpha_low');call make_sparse_spd_matrix(x,alpha=-.1_dp)
    case('alpha_high');call make_sparse_spd_matrix(x,alpha=1.1_dp)
    case('alpha_nan');call make_sparse_spd_matrix(x,alpha=nan)
    case('alpha_inf');call make_sparse_spd_matrix(x,alpha=inf)
    case('coef_low');call make_sparse_spd_matrix(x,smallest_coef=-.1_dp)
    case('coef_high');call make_sparse_spd_matrix(x,largest_coef=1.1_dp)
    case('coef_nan');call make_sparse_spd_matrix(x,smallest_coef=nan)
    case('coef_inf');call make_sparse_spd_matrix(x,largest_coef=inf)
    case('spd_seed');call make_sparse_spd_matrix(x,random_state=-1)
    end select
    error stop 'invalid structured dataset input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) kind,n,k,p,s,seed,normalized,alpha,low,high
    write(*,'(a,1x,i0)') 'BEGIN',index
    if(kind==0) then
      call make_sparse_coded_signal(n,k,p,s,x,d,code,seed)
      call make_sparse_coded_signal(n,k,p,s,other,d2,c2,seed)
      call matrix('data',x);call matrix('dictionary',d);call matrix('code',code)
      call vector('metadata',real([size(x,1),size(x,2),size(d,1),size(d,2),size(code,1),size(code,2)],dp))
      call vector('support',real(count(code/=0,dim=2),dp))
      call vector('repeat_identity',[merge(1._dp,0._dp,all(x==other).and.all(d==d2).and.all(code==c2))])
    else
      call make_sparse_spd_matrix(x,n,alpha,normalized==1,low,high,seed)
      call make_sparse_spd_matrix(csr,n,alpha,normalized==1,low,high,seed)
      other=csr%to_dense();call matrix('matrix',x);call matrix('csr_dense',other)
      call vector('metadata',real([size(x,1),size(x,2)],dp))
      call vector('csr_consistency',[merge(1._dp,0._dp,all(x==other))])
      call vector('csr_nonzeros',[real(csr%nnz(),dp)])
    end if
    write(*,'(a)') 'END'
  end do
contains
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    integer :: i
    write(*,'(a,1x,i0)',advance='no') name,size(a)
    do i=1,size(a)
      write(*,'(1x,es25.17e3)',advance='no') a(i)
    end do
    write(*,*)
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(transpose(a),[size(a)]))
  end subroutine
end program
