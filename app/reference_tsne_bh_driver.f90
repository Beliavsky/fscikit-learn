program reference_tsne_bh_driver
  use sklearn_common, only: dp
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_tsne_core, only: tsne_gradient_descent
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf
  use sklearn_tsne_bh, only: tsne_bh_objective,tsne_bh_probabilities
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(csr_matrix) :: graph,p
  real(dp), allocatable :: x(:,:),gradient(:,:),values(:,:)
  integer, allocatable :: cells(:,:)
  real(dp), allocatable :: trace(:,:),empty(:)
  real(dp) :: angle,error,perplexity,rate,momentum,min_norm
  integer :: steps,first,patience,iteration,single,auto_rate
  character(40) :: argument
  integer :: mode,n,d,degrees,skip,compute,nnz
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument));error stop 'invalid Barnes-Hut input accepted'
  end if
  read(*,*) mode,n,d,degrees,angle,skip,compute,perplexity,nnz
  graph%n_rows=n;graph%n_cols=n
  allocate(graph%indptr(n+1),graph%indices(nnz),graph%data(nnz))
  read(*,*) graph%indptr;read(*,*) graph%indices;read(*,*) graph%data
  if(mode==0) then
    call tsne_bh_probabilities(graph,perplexity,p)
    call integers(reshape(int(p%indptr,int64),[n+1,1]));call integers(reshape(int(p%indices,int64),[size(p%indices),1]))
    call matrix(reshape(p%data,[size(p%data),1]))
  else
    allocate(x(n,d));read(*,*) x
    if(mode==1) then
      call tsne_bh_objective(x,graph,degrees,angle,error,gradient,skip,compute/=0,cells,values)
      call matrix(reshape([error],[1,1]));call matrix(gradient);call integers(int(cells,int64));call matrix(values)
    else
      read(*,*) steps,first,patience,rate,momentum,min_norm,single,auto_rate
      allocate(empty(0))
      call tsne_gradient_descent(x,empty,degrees,first,first+steps,patience,momentum,rate,min_norm,single/=0,auto_rate/=0, &
        error,iteration,trace,graph,angle)
      call matrix(x);call matrix(reshape([error],[1,1]));call integers(reshape(int([iteration],int64),[1,1]));call matrix(trace)
    end if
  end if
contains
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: z(4,2),distance(4,4),angle,error,perplexity
    real(dp), allocatable :: gradient(:,:)
    integer :: degrees,skip
    type(csr_matrix) :: q,result
    z=0;distance=.1_dp;angle=.5_dp;perplexity=2;degrees=1;skip=0;q=csr_from_dense(distance,.true.)
    select case(name)
    case('angle_nan');angle=ieee_value(0._dp,ieee_quiet_nan)
    case('angle_inf');angle=ieee_value(0._dp,ieee_positive_inf)
    case('angle_negative');angle=-1
    case('angle_large');angle=1.1_dp
    case('degrees');degrees=0
    case('skip_negative');skip=-1
    case('skip_large');skip=5
    case('probability_negative');q%data(1)=-1
    case('probability_nan');q%data(1)=ieee_value(0._dp,ieee_quiet_nan)
    case('probability_range');q%data(1)=huge(1._dp)
    case('probability_shape');q=csr_from_dense(distance(:3,:3),.true.)
    case('position_range');z=huge(1._dp)/2
    case('dimension');call tsne_bh_objective(distance,q,1,angle,error,gradient);return
    case('perplexity_zero');call tsne_bh_probabilities(q,0._dp,result);return
    case('perplexity_large');call tsne_bh_probabilities(q,4._dp,result);return
    case('distance_negative');q%data(1)=-1;call tsne_bh_probabilities(q,perplexity,result);return
    case('distance_inf');q%data(1)=ieee_value(0._dp,ieee_positive_inf)
      call tsne_bh_probabilities(q,perplexity,result);return
    case('distance_range');q%data=huge(1._dp);call tsne_bh_probabilities(q,perplexity,result);return
    case('distance_shape');q=csr_from_dense(distance(:,:3),.true.);call tsne_bh_probabilities(q,perplexity,result);return
    case default;error stop 'unknown Barnes-Hut backend rejection'
    end select
    call tsne_bh_objective(z,q,degrees,angle,error,gradient,skip)
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(16(i0,1x))',a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(8(es25.16e3,1x))',a
  end subroutine
end program
