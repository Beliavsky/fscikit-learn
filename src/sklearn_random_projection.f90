! Gaussian and sparse random projections, with optional CSR component storage.
! Adapted from sklearn 1.6.1 random_projection.py; see THIRD_PARTY_NOTICES.md.
module sklearn_random_projection
  use sklearn_sparse, only: csr_matrix,csr_from_dense
  use sklearn_common
  use sklearn_ieee_math, only: ieee_quotient,ieee_difference,ieee_product
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_random_binomial, only: legacy_binomial
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan
  implicit none
  private
  public :: gaussian_random_projection,sparse_random_projection,johnson_lindenstrauss_min_dim
  ! Keep only sampled columns until the subsequent sign-draw phase.  A rectangular
  ! n_features by n_components workspace defeats low-density sampling.
  type :: sampled_columns
    integer, allocatable :: values(:)
  end type
  type :: gaussian_random_projection
    integer :: n_components=0,n_components_=0,random_state=0,n_features_in_=0
    real(dp) :: eps=0.1_dp
    logical :: compute_inverse_components=.false.
    real(dp), allocatable :: components_(:,:),inverse_components_(:,:)
  contains
    procedure, private :: fit_dense=>projection_fit
    procedure, private :: fit_csr=>projection_fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure, private :: make_components=>gaussian_components
    procedure, private :: transform_dense=>projection_transform
    procedure, private :: transform_csr=>projection_transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure, private :: fit_transform_dense=>projection_fit_transform
    procedure, private :: fit_transform_csr=>projection_fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
    procedure, private :: inverse_dense=>projection_inverse
    procedure, private :: inverse_csr=>projection_inverse_csr
    generic :: inverse_transform=>inverse_dense,inverse_csr
  end type
  type, extends(gaussian_random_projection) :: sparse_random_projection
    real(dp) :: density=-1,density_=0
    ! Native storage adapter; the default retains the original dense-array API.
    logical :: store_sparse_components=.false.
    type(csr_matrix) :: components_csr_
  contains
    procedure, private :: make_components=>sparse_components
    procedure :: transform_sparse=>projection_sparse_output
    procedure :: fit_transform_sparse=>projection_fit_sparse_output
  end type
contains
  integer function johnson_lindenstrauss_min_dim(n_samples,eps) result(n)
    integer, intent(in) :: n_samples
    real(dp), intent(in) :: eps
    real(dp) :: bound
    call require(ieee_is_finite(eps),'Johnson-Lindenstrauss eps must be finite')
    call require(n_samples>0.and.eps>0.and.eps<1,'invalid Johnson-Lindenstrauss parameters')
    bound=4*log(real(n_samples,dp))/(eps**2/2-eps**3/3)
    call require(bound<real(huge(n),dp),'Johnson-Lindenstrauss dimension exceeds integer range')
    n=int(bound)
  end function
  subroutine projection_setup(self,n_samples,n_features)
    class(gaussian_random_projection), intent(inout) :: self
    integer, intent(in) :: n_samples,n_features
    call require(self%n_components>=0,'projection n_components must be nonnegative')
    call require(ieee_is_finite(self%eps),'projection eps must be finite')
    call require(self%eps>0,'projection eps must be positive')
    call require(self%random_state>=0,'projection random_state must be nonnegative')
    self%n_features_in_=n_features
    self%n_components_=self%n_components
    if(self%n_components==0) then
      self%n_components_=johnson_lindenstrauss_min_dim(n_samples,self%eps)
      call require(self%n_components_>0.and.self%n_components_<=n_features, &
        'automatic projection dimension must be between 1 and n_features')
    end if
    if(allocated(self%inverse_components_)) deallocate(self%inverse_components_)
  end subroutine
  function pseudoinverse(a) result(inv)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: inv(size(a,2),size(a,1)),u(size(a,1),size(a,2))
    real(dp) :: s(size(a,2)),v(size(a,2),size(a,2)),weights(size(a,2)),cutoff
    call dense_svd(a,s,u,v); cutoff=1.e-15_dp*maxval(s); weights=0
    where(s>cutoff) weights=1/max(s,tiny(1._dp))
    inv=matmul(v*spread(weights,1,size(a,2)),transpose(u))
  end function
  subroutine projection_fit(self,x)
    class(gaussian_random_projection), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    call check_x(x);call projection_setup(self,size(x,1),size(x,2))
    call self%make_components()
  end subroutine
  subroutine check_projection_csr(x)
    type(csr_matrix), intent(in) :: x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'projection CSR input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'projection CSR input contains NaN or infinity')
  end subroutine
  subroutine projection_fit_csr(self,x)
    class(gaussian_random_projection), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    call check_projection_csr(x);call projection_setup(self,x%n_rows,x%n_cols)
    call self%make_components()
  end subroutine
  subroutine gaussian_components(self)
    class(gaussian_random_projection), intent(inout) :: self
    type(legacy_random_state) :: rng
    real(dp) :: scale
    integer :: i,j
    call rng%seed(self%random_state)
    if(allocated(self%components_)) deallocate(self%components_)
    allocate(self%components_(self%n_components_,self%n_features_in_))
    scale=1/sqrt(real(self%n_components_,dp))
    do i=1,self%n_components_
      do j=1,self%n_features_in_
        self%components_(i,j)=scale*rng%normal()
      end do
    end do
    if(self%compute_inverse_components) self%inverse_components_=pseudoinverse(self%components_)
  end subroutine
  subroutine sparse_components(self)
    class(sparse_random_projection), intent(inout) :: self
    type(legacy_random_state) :: rng
    real(dp) :: magnitude
    integer :: i,j,p,sign,offset
    integer(int64) :: total
    type(sampled_columns), allocatable :: indices(:)
    call rng%seed(self%random_state)
    call require(ieee_is_finite(self%density),'projection density must be finite')
    call require(self%density==-1.or.(self%density>0.and.self%density<=1),'projection density must be in (0,1] or -1')
    self%density_=self%density;p=self%n_features_in_
    if(self%density==-1) self%density_=1/sqrt(real(p,dp))
    if(allocated(self%components_)) deallocate(self%components_)
    self%components_csr_=csr_matrix()
    if(.not.self%store_sparse_components) then
      allocate(self%components_(self%n_components_,p));self%components_=0
    end if
    if(self%density_==1) then
      if(self%store_sparse_components) then
        total=int(self%n_components_,int64)*p
        call allocate_components(self,total)
        self%components_csr_%indptr=[(1+(i-1)*p,i=1,self%n_components_+1)]
      end if
      magnitude=1/sqrt(real(self%n_components_,dp))
      offset=1
      do i=1,self%n_components_
        do j=1,p
          sign=2*legacy_binomial(1,.5_dp,rng)-1
          if(self%store_sparse_components) then
            self%components_csr_%indices(offset)=j
            self%components_csr_%data(offset)=magnitude*sign;offset=offset+1
          else
            self%components_(i,j)=magnitude*sign
          end if
        end do
      end do
    else
      allocate(indices(self%n_components_))
      do i=1,self%n_components_
        j=legacy_binomial(p,self%density_,rng)
        indices(i)%values=sample_indices(p,j,rng)
      end do
      if(self%store_sparse_components) then
        total=0
        do i=1,self%n_components_
          total=total+size(indices(i)%values,kind=int64)
        end do
        call allocate_components(self,total)
      end if
      magnitude=sqrt(ieee_quotient(1._dp,self%density_))/sqrt(real(self%n_components_,dp))
      offset=1
      do i=1,self%n_components_
        do j=1,size(indices(i)%values)
          sign=2*legacy_binomial(1,.5_dp,rng)-1
          if(self%store_sparse_components) then
            self%components_csr_%indices(offset)=indices(i)%values(j)
            self%components_csr_%data(offset)=magnitude*sign;offset=offset+1
          else
            self%components_(i,indices(i)%values(j))=magnitude*sign
          end if
        end do
        if(self%store_sparse_components) self%components_csr_%indptr(i+1)=offset
        deallocate(indices(i)%values)
      end do
    end if
    if(self%compute_inverse_components) self%inverse_components_=component_inverse(self)
  end subroutine
  subroutine allocate_components(self,total)
    class(sparse_random_projection), intent(inout) :: self
    integer(int64), intent(in) :: total
    call require(total<int(huge(1),int64),'projection CSR components exceed integer range')
    self%components_csr_%n_rows=self%n_components_;self%components_csr_%n_cols=self%n_features_in_
    allocate(self%components_csr_%indptr(self%n_components_+1), &
      self%components_csr_%indices(int(total)),self%components_csr_%data(int(total)))
    self%components_csr_%indptr(1)=1
  end subroutine
  logical function sparse_components_fitted(self) result(fitted)
    class(gaussian_random_projection), intent(in) :: self
    fitted=.false.
    select type(self)
    class is(sparse_random_projection)
      fitted=allocated(self%components_csr_%indptr)
    end select
  end function
  subroutine check_fitted(self)
    class(gaussian_random_projection), intent(in) :: self
    call require(allocated(self%components_).or.sparse_components_fitted(self),'random projection is not fitted')
  end subroutine
  function component_inverse(self) result(a)
    class(gaussian_random_projection), intent(in) :: self
    real(dp), allocatable :: a(:,:)
    call check_fitted(self)
    if(sparse_components_fitted(self)) then
      select type(self)
      class is(sparse_random_projection);a=pseudoinverse(self%components_csr_%to_dense())
      end select
    else
      a=pseudoinverse(self%components_)
    end if
  end function
  function component_columns(self) result(columns)
    class(gaussian_random_projection), intent(in) :: self
    type(csr_matrix) :: columns
    integer, allocatable :: next(:)
    integer :: i,j,k,offset
    call check_fitted(self)
    if(.not.sparse_components_fitted(self)) then
      columns=csr_from_dense(transpose(self%components_));return
    end if
    select type(self)
    class is(sparse_random_projection)
      associate(a=>self%components_csr_)
        columns%n_rows=a%n_cols;columns%n_cols=a%n_rows
        allocate(columns%indptr(a%n_cols+1),columns%indices(a%nnz()),columns%data(a%nnz()))
        allocate(next(a%n_cols));next=0
        do k=1,a%nnz()
          next(a%indices(k))=next(a%indices(k))+1
        end do
        columns%indptr(1)=1
        do j=1,a%n_cols
          columns%indptr(j+1)=columns%indptr(j)+next(j)
        end do
        next=columns%indptr(:a%n_cols)
        do i=1,a%n_rows
          do k=a%indptr(i),a%indptr(i+1)-1
            j=a%indices(k);offset=next(j);next(j)=offset+1
            columns%indices(offset)=i;columns%data(offset)=a%data(k)
          end do
        end do
      end associate
    end select
  end function
  function sample_indices(n,k,rng) result(selected)
    integer, intent(in) :: n,k
    type(legacy_random_state), intent(inout) :: rng
    integer, allocatable :: selected(:),pool(:),seen(:)
    real(dp) :: ratio
    integer :: i,j,swap,bucket
    allocate(selected(k));ratio=real(k,dp)/n
    if(ratio>.01_dp.and.ratio<.99_dp) then
      pool=[(i,i=1,n)]
      do i=n,2,-1
        j=rng%integer_index(i);swap=pool(i);pool(i)=pool(j);pool(j)=swap
      end do
      selected=pool(:k)
    else if(ratio<.2_dp) then
      ! sklearn's tracking-selection branch uses set membership.  Linear scans
      ! of the selected prefix cost O(k**2), even at a very low density.
      allocate(seen(2*k+1));seen=0
      do i=1,k
        do
          j=rng%integer_index(n)
          bucket=int(modulo(int(j,int64)*2654435761_int64,int(size(seen),int64)))+1
          do while(seen(bucket)/=0)
            if(seen(bucket)==j) exit
            bucket=modulo(bucket,size(seen))+1
          end do
          if(seen(bucket)==0) exit
        end do
        seen(bucket)=j
        selected(i)=j
      end do
    else
      selected=[(i,i=1,k)]
      do i=k+1,n
        j=rng%integer_index(i)
        if(j<=k) selected(j)=i
      end do
    end if
  end function
  function projection_transform(self,x) result(z)
    class(gaussian_random_projection), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    integer :: i,j,k,column
    call check_fitted(self)
    if(sparse_components_fitted(self)) then
      select type(self)
      class is(sparse_random_projection)
        call check_xy(transpose(x),self%components_csr_%n_cols)
        allocate(z(size(x,1),self%components_csr_%n_rows));z=0
        do j=1,self%components_csr_%n_rows
          do i=1,size(x,1)
            do k=self%components_csr_%indptr(j),self%components_csr_%indptr(j+1)-1
              column=self%components_csr_%indices(k)
              z(i,j)=ieee_difference(z(i,j),-ieee_product(x(i,column),self%components_csr_%data(k)))
            end do
          end do
        end do
      end select
      return
    end if
    call check_xy(transpose(x),size(self%components_,2))
    z=matmul(x,transpose(self%components_))
  end function
  function projection_transform_csr(self,x) result(z)
    class(gaussian_random_projection), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: z(:,:)
    type(csr_matrix) :: columns
    integer :: i,j,k,entry
    call check_fitted(self)
    call check_projection_csr(x)
    if(sparse_components_fitted(self)) then
      columns=component_columns(self)
      call require(x%n_cols==columns%n_rows,'projection CSR feature count mismatch')
      allocate(z(x%n_rows,columns%n_cols));z=0
      do i=1,x%n_rows
        do k=x%indptr(i),x%indptr(i+1)-1
          do entry=columns%indptr(x%indices(k)),columns%indptr(x%indices(k)+1)-1
            j=columns%indices(entry)
            z(i,j)=ieee_difference(z(i,j),-ieee_product(x%data(k),columns%data(entry)))
          end do
        end do
      end do
      return
    end if
    call require(x%n_cols==size(self%components_,2),'projection CSR feature count mismatch')
    z=x%matmul(transpose(self%components_))
  end function
  function projection_sparse_output(self,x) result(z)
    class(sparse_random_projection), intent(in) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: z,components
    real(dp), allocatable :: row(:)
    integer(int64) :: total
    integer :: i,j,k,entry,pass,offset,nz
    call check_fitted(self)
    call check_projection_csr(x)
    components=component_columns(self)
    call require(x%n_cols==components%n_rows,'projection CSR feature count mismatch')
    allocate(row(components%n_cols))
    z%n_rows=x%n_rows;z%n_cols=size(row);allocate(z%indptr(z%n_rows+1));z%indptr(1)=1
    total=0
    do pass=1,2
      offset=1
      do i=1,x%n_rows
        row=0
        do k=x%indptr(i),x%indptr(i+1)-1
          do entry=components%indptr(x%indices(k)),components%indptr(x%indices(k)+1)-1
            j=components%indices(entry)
            row(j)=ieee_difference(row(j),-ieee_product(x%data(k),components%data(entry)))
          end do
        end do
        nz=0
        do j=1,size(row)
          if(.not.ieee_is_nan(row(j))) then
            if(row(j)==0) cycle
          end if
          nz=nz+1
          if(pass==2) then
            z%indices(offset)=j;z%data(offset)=row(j);offset=offset+1
          end if
        end do
        if(pass==1) then
          total=total+nz
          call require(total<int(huge(1),int64),'projection CSR output exceeds integer range')
          z%indptr(i+1)=int(total)+1
        end if
      end do
      if(pass==1) allocate(z%indices(int(total)),z%data(int(total)))
    end do
  end function
  function projection_fit_sparse_output(self,x) result(z)
    class(sparse_random_projection), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    type(csr_matrix) :: z
    call self%fit(x);z=self%transform_sparse(x)
  end function
  function projection_fit_transform_csr(self,x) result(z)
    class(gaussian_random_projection), intent(inout) :: self
    type(csr_matrix), intent(in) :: x
    real(dp), allocatable :: z(:,:)
    call self%fit(x);z=self%transform(x)
  end function
  function projection_inverse_csr(self,z) result(x)
    class(gaussian_random_projection), intent(in) :: self
    type(csr_matrix), intent(in) :: z
    real(dp), allocatable :: x(:,:)
    call check_fitted(self)
    call check_projection_csr(z)
    if(sparse_components_fitted(self)) then
      select type(self)
      class is(sparse_random_projection)
        call require(z%n_cols==self%components_csr_%n_rows,'projection CSR latent count mismatch')
      end select
    else
      call require(z%n_cols==size(self%components_,1),'projection CSR latent count mismatch')
    end if
    if(allocated(self%inverse_components_)) then
      x=z%matmul(transpose(self%inverse_components_))
    else
      x=z%matmul(transpose(component_inverse(self)))
    end if
  end function
  function projection_fit_transform(self,x) result(z)
    class(gaussian_random_projection), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
  function projection_inverse(self,z) result(x)
    class(gaussian_random_projection), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:)
    call check_fitted(self)
    if(sparse_components_fitted(self)) then
      select type(self)
      class is(sparse_random_projection)
        call check_xy(transpose(z),self%components_csr_%n_rows)
      end select
    else
      call check_xy(transpose(z),size(self%components_,1))
    end if
    if(allocated(self%inverse_components_)) then
      x=matmul(z,transpose(self%inverse_components_))
    else
      x=matmul(z,transpose(component_inverse(self)))
    end if
  end function
end module
