program reference_pairwise_measures_driver
  use sklearn_common, only: dp
  use sklearn_metrics, only: cosine_distances,nan_euclidean_distances,haversine_distances, &
    paired_euclidean_distances,paired_manhattan_distances,paired_cosine_distances,paired_distances, &
    laplacian_kernel,additive_chi2_kernel,chi2_kernel
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_ieee_math, only: ieee_difference
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_positive_inf,ieee_is_nan
  implicit none
  real(dp), allocatable :: x(:,:),y(:,:),before(:,:),other(:,:)
  type(csr_matrix) :: sx,sy
  integer :: cases,index,kind,n,m,p,i
  real(dp) :: gamma,marker,nan,inf
  character(60) :: mode
  nan=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
  call get_command_argument(1,mode)
  if(len_trim(mode)>0) then
    call reject(trim(mode));error stop 'invalid pairwise input accepted'
  end if
  read(*,*) cases
  do index=1,cases
    read(*,*) kind,n,m,p,gamma,marker
    allocate(x(n,p),y(m,p))
    do i=1,n
      read(*,*) x(i,:)
    end do
    do i=1,m
      read(*,*) y(i,:)
    end do
    before=x;other=y
    write(*,'(a,1x,i0)') 'BEGIN',index
    select case(kind)
    case(0)
      sx=csr_from_dense(x);sy=csr_from_dense(y)
      call matrix('cosine',cosine_distances(x,y));call matrix('cosine_self',cosine_distances(x))
      call matrix('cosine_ss',cosine_distances(sx,sy));call matrix('cosine_sd',cosine_distances(sx,y))
      call matrix('cosine_ds',cosine_distances(x,sy));call matrix('cosine_sparse_self',cosine_distances(sx))
      call matrix('laplacian_default',laplacian_kernel(x,y));call matrix('laplacian',laplacian_kernel(x,y,gamma))
      call matrix('laplacian_self',laplacian_kernel(x));call matrix('laplacian_sparse_self',laplacian_kernel(sx))
      call matrix('laplacian_ss',laplacian_kernel(sx,sy,gamma))
      call matrix('laplacian_sd',laplacian_kernel(sx,y,gamma));call matrix('laplacian_ds',laplacian_kernel(x,sy,gamma))
      if(n==m) then
        call vector('euclidean',paired_euclidean_distances(x,y))
        call vector('euclidean_ss',paired_euclidean_distances(sx,sy))
        call vector('euclidean_sd',paired_euclidean_distances(sx,y))
        call vector('euclidean_ds',paired_euclidean_distances(x,sy))
        call vector('manhattan',paired_manhattan_distances(x,y))
        call vector('manhattan_ss',paired_manhattan_distances(sx,sy))
        call vector('manhattan_sd',paired_manhattan_distances(sx,y))
        call vector('manhattan_ds',paired_manhattan_distances(x,sy))
        call vector('paired_cosine',paired_cosine_distances(x,y))
        call vector('paired_cosine_ss',paired_cosine_distances(sx,sy))
        call vector('paired_cosine_sd',paired_cosine_distances(sx,y))
        call vector('paired_cosine_ds',paired_cosine_distances(x,sy))
        call vector('default_paired',paired_distances(x,y))
        call vector('alias_l2',paired_distances(x,y,'l2'));call vector('alias_l1',paired_distances(x,y,'l1'))
        call vector('alias_cityblock',paired_distances(x,y,'cityblock'))
        call vector('dispatch_euclidean',paired_distances(x,y,'euclidean'))
        call vector('dispatch_manhattan',paired_distances(x,y,'manhattan'))
        call vector('dispatch_cosine',paired_distances(x,y,'cosine'))
        call vector('callback',paired_distances(x,y,chebyshev))
      end if
    case(1)
      call matrix('nan_squared',nan_euclidean_distances(x,y,squared=.true.,missing_values=marker))
      call matrix('nan_root',nan_euclidean_distances(x,y,missing_values=marker))
      call matrix('nan_self_squared',nan_euclidean_distances(x,squared=.true.,missing_values=marker))
      call matrix('nan_self_root',nan_euclidean_distances(x,missing_values=marker))
      if(ieee_is_nan(marker)) call matrix('nan_default',nan_euclidean_distances(x,y))
    case(2)
      call matrix('additive_chi2',additive_chi2_kernel(x,y));call matrix('additive_chi2_self',additive_chi2_kernel(x))
      call matrix('chi2_default',chi2_kernel(x,y));call matrix('chi2',chi2_kernel(x,y,gamma))
      call matrix('chi2_self',chi2_kernel(x));call matrix('chi2_self_gamma',chi2_kernel(x,gamma=gamma))
    case(3)
      sx=csr_from_dense(x);sy=csr_from_dense(y)
      call matrix('haversine',haversine_distances(x,y));call matrix('haversine_self',haversine_distances(x))
      call matrix('haversine_ss',haversine_distances(sx,sy));call matrix('haversine_sparse_self',haversine_distances(sx))
      call matrix('haversine_sd',haversine_distances(sx,y));call matrix('haversine_ds',haversine_distances(x,sy))
    end select
    call vector('input_preserved',[merge(1._dp,0._dp,equal(x,before).and.equal(y,other))])
    write(*,'(a)') 'END'
    deallocate(x,y)
  end do
contains
  logical function equal(a,b) result(ok)
    real(dp), intent(in) :: a(:,:),b(:,:)
    integer :: i,j
    ok=.true.
    do i=1,size(a,1)
      do j=1,size(a,2)
        if(ieee_is_nan(a(i,j))) then
          if(.not.ieee_is_nan(b(i,j))) ok=.false.
        else if(a(i,j)/=b(i,j)) then
          ok=.false.
        end if
      end do
    end do
  end function
  real(dp) function chebyshev(a,b) result(value)
    real(dp), intent(in) :: a(:),b(:)
    integer :: i
    value=0
    do i=1,size(a)
      value=max(value,abs(ieee_difference(a(i),b(i))))
    end do
  end function
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    integer :: j
    write(*,'(a,1x,i0)',advance='no') name,size(a)
    do j=1,size(a)
      write(*,'(1x,es25.17e3)',advance='no') a(j)
    end do
    write(*,*)
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(transpose(a),[size(a)]))
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    real(dp) :: a(2,2),b(2,2)
    real(dp), allocatable :: result_matrix(:,:),result_vector(:)
    type(csr_matrix) :: bad
    a=1;b=2;bad=csr_from_dense(a)
    select case(name)
    case('empty');result_matrix=cosine_distances(reshape([real(dp)::],[0,2]))
    case('features');result_matrix=laplacian_kernel(a,b(:,:1))
    case('paired_samples');result_vector=paired_euclidean_distances(a,b(:1,:))
    case('nan_input');a(1,1)=nan;result_matrix=cosine_distances(a,b)
    case('inf_input');a(1,1)=inf;result_vector=paired_cosine_distances(a,b)
    case('paired_metric');result_vector=paired_distances(a,b,'invalid')
    case('haversine_shape');result_matrix=haversine_distances(a(:,:1))
    case('lap_gamma_zero');result_matrix=laplacian_kernel(a,gamma=0._dp)
    case('lap_gamma_negative');result_matrix=laplacian_kernel(a,gamma=-1._dp)
    case('lap_gamma_nan');result_matrix=laplacian_kernel(a,gamma=nan)
    case('lap_gamma_inf');result_matrix=laplacian_kernel(a,gamma=inf)
    case('chi_gamma_zero');result_matrix=chi2_kernel(a,gamma=0._dp)
    case('chi_gamma_negative');result_matrix=chi2_kernel(a,gamma=-1._dp)
    case('chi_gamma_nan');result_matrix=chi2_kernel(a,gamma=nan)
    case('chi_gamma_inf');result_matrix=chi2_kernel(a,gamma=inf)
    case('chi_negative_x');a(1,1)=-1;result_matrix=additive_chi2_kernel(a,b)
    case('chi_negative_y');b(1,1)=-1;result_matrix=chi2_kernel(a,b)
    case('nan_inf');a(1,1)=inf;result_matrix=nan_euclidean_distances(a)
    case('marker_nan_input');a(1,1)=nan;result_matrix=nan_euclidean_distances(a,missing_values=-99._dp)
    case('nan_features');result_matrix=nan_euclidean_distances(a,b(:,:1))
    case('nan_empty');result_matrix=nan_euclidean_distances(reshape([real(dp)::],[0,2]))
    case('csr_nan');bad%data(1)=nan;result_matrix=cosine_distances(bad)
    case('csr_inf');bad%data(1)=inf;result_matrix=laplacian_kernel(bad)
    case('csr_duplicate');bad%indices(2)=bad%indices(1);result_vector=paired_euclidean_distances(bad,b)
    case('csr_unsorted');bad%indices(:2)=[2,1];result_matrix=haversine_distances(bad)
    case('csr_index');bad%indices(1)=3;result_matrix=cosine_distances(bad)
    end select
  end subroutine
end program
