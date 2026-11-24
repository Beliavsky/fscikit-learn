! sklearn 1.6.1 sample generators and SciPy 1.15.3 sparse random contracts (BSD).
! Original native interfaces/workspaces are MIT; see THIRD_PARTY_NOTICES.md.
module sklearn_structured_datasets
  use sklearn_common, only: dp,require
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  private
  public :: make_sparse_coded_signal,make_sparse_spd_matrix
  interface make_sparse_spd_matrix
    module procedure sparse_spd_dense,sparse_spd_csr
  end interface
contains
  function permutation(n,rng) result(order)
    integer, intent(in) :: n
    type(legacy_random_state), intent(inout) :: rng
    integer :: order(n),i,j,t
    order=[(i,i=1,n)]
    do i=n,2,-1
      j=rng%integer_index(i);t=order(i);order(i)=order(j);order(j)=t
    end do
  end function
  subroutine make_sparse_coded_signal(n_samples,n_components,n_features,n_nonzero_coefs,data,dictionary,code,random_state)
    integer, intent(in) :: n_samples,n_components,n_features,n_nonzero_coefs
    real(dp), allocatable, intent(out) :: data(:,:),dictionary(:,:),code(:,:)
    integer, optional, intent(in) :: random_state
    type(legacy_random_state) :: rng
    integer, allocatable :: order(:)
    integer :: i,j,seed
    call require(n_samples>=1,'sparse coded n_samples must be positive')
    call require(n_components>=1,'sparse coded n_components must be positive')
    call require(n_features>=1,'sparse coded n_features must be positive')
    call require(n_nonzero_coefs>=1.and.n_nonzero_coefs<=n_components, &
      'sparse coded n_nonzero_coefs must be between one and n_components')
    seed=0;if(present(random_state)) seed=random_state
    call rng%seed(seed)
    allocate(dictionary(n_components,n_features),code(n_samples,n_components));code=0
    ! Source draws a (features, components) C-order array before transposing.
    do j=1,n_features
      do i=1,n_components
        dictionary(i,j)=rng%normal()
      end do
    end do
    do i=1,n_components
      dictionary(i,:)=dictionary(i,:)/sqrt(sum(dictionary(i,:)**2))
    end do
    do i=1,n_samples
      order=permutation(n_components,rng)
      do j=1,n_nonzero_coefs
        code(i,order(j))=rng%normal()
      end do
    end do
    data=matmul(code,dictionary)
  end subroutine
  subroutine sparse_spd_dense(matrix,n_dim,alpha,norm_diag,smallest_coef,largest_coef,random_state)
    real(dp), allocatable, intent(out) :: matrix(:,:)
    integer, optional, intent(in) :: n_dim,random_state
    real(dp), optional, intent(in) :: alpha,smallest_coef,largest_coef
    logical, optional, intent(in) :: norm_diag
    type(legacy_random_state) :: rng
    real(dp), allocatable :: chol(:,:),scales(:)
    integer, allocatable :: order(:),axis(:)
    integer :: n,seed,total,count,i,r,c
    real(dp) :: a,low,high,expected,value
    logical :: normalize
    n=1;seed=0;a=.95_dp;low=.1_dp;high=.9_dp;normalize=.false.
    if(present(n_dim)) n=n_dim
    if(present(random_state)) seed=random_state
    if(present(alpha)) a=alpha
    if(present(smallest_coef)) low=smallest_coef
    if(present(largest_coef)) high=largest_coef
    if(present(norm_diag)) normalize=norm_diag
    call require(n>=1,'sparse SPD n_dim must be positive')
    call require(int(n,int64)*n<=int(huge(total),int64),'sparse SPD population exceeds index range')
    call require(ieee_is_finite(a),'sparse SPD alpha must be finite')
    call require(a>=0.and.a<=1,'sparse SPD alpha must be in [0,1]')
    call require(ieee_is_finite(low).and.ieee_is_finite(high),'sparse SPD coefficients must be finite')
    call require(low>=0.and.low<=1.and.high>=0.and.high<=1,'sparse SPD coefficients must be in [0,1]')
    ! NumPy uniform permits reversed bounds; retain its signed interval.
    call rng%seed(seed);total=n*n;expected=(1-a)*real(total,dp)
    count=int(floor(expected))
    if(expected-count>.5_dp.or.(expected-count==.5_dp.and.mod(count,2)==1)) count=count+1
    ! Legacy RandomState.choice(replace=False) permutes the whole population,
    ! even when the requested size is zero. SciPy indexes that population in F order.
    order=permutation(total,rng);allocate(chol(n,n));chol=0
    do i=1,count
      value=low+(high-low)*rng%uniform();r=mod(order(i)-1,n)+1;c=(order(i)-1)/n+1
      if(r>c) chol(r,c)=value
    end do
    axis=permutation(n,rng)
    ! sklearn permutes rows, transposes, then permutes rows again.
    chol=transpose(chol(axis,axis))
    do i=1,n
      chol(i,i)=-1
    end do
    matrix=matmul(transpose(chol),chol)
    if(normalize) then
      allocate(scales(n))
      do i=1,n
        scales(i)=1/sqrt(matrix(i,i))
      end do
      ! Sparse source scales rows before columns.
      matrix=matrix*spread(scales,2,n);matrix=matrix*spread(scales,1,n)
    end if
  end subroutine
  subroutine sparse_spd_csr(matrix,n_dim,alpha,norm_diag,smallest_coef,largest_coef,random_state)
    type(csr_matrix), intent(out) :: matrix
    integer, optional, intent(in) :: n_dim,random_state
    real(dp), optional, intent(in) :: alpha,smallest_coef,largest_coef
    logical, optional, intent(in) :: norm_diag
    real(dp), allocatable :: dense(:,:)
    call sparse_spd_dense(dense,n_dim,alpha,norm_diag,smallest_coef,largest_coef,random_state)
    matrix=csr_from_dense(dense)
  end subroutine
end module
