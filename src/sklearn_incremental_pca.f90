! IncrementalPCA update follows sklearn 1.6.1 decomposition/_incremental_pca.py.
! Upstream BSD notice: LICENSES/scikit-learn-COPYING; original contributions MIT.
module sklearn_incremental_pca
  use sklearn_common, only: dp,require,check_x,dense_svd
  use sklearn_sparse, only: csr_matrix
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: incremental_pca
  type :: incremental_pca
    integer :: n_components=0,batch_size=0,n_components_=0,batch_size_=0,n_samples_seen_=0,n_features_in_=0
    logical :: whiten=.false.
    real(dp) :: noise_variance_=0
    real(dp), allocatable :: components_(:,:),mean_(:),var_(:),singular_values_(:)
    real(dp), allocatable :: explained_variance_(:),explained_variance_ratio_(:)
  contains
    procedure :: fit_dense=>ipca_fit,fit_csr
    generic :: fit=>fit_dense,fit_csr
    procedure :: partial_fit_dense=>ipca_partial,partial_fit_csr
    generic :: partial_fit=>partial_fit_dense,partial_fit_csr
    procedure :: transform_dense=>ipca_transform,transform_csr
    generic :: transform=>transform_dense,transform_csr
    procedure :: inverse_transform=>ipca_inverse
    procedure :: fit_transform_dense=>ipca_fit_transform,fit_transform_csr
    generic :: fit_transform=>fit_transform_dense,fit_transform_csr
  end type
contains
  subroutine check_csr(x)
    type(csr_matrix),intent(in)::x
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'IncrementalPCA sparse input must be nonempty')
    call require(all(ieee_is_finite(x%data)),'IncrementalPCA sparse input must be finite')
  end subroutine

  function dense_batch(x,first,last)result(batch)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::first,last
    real(dp),allocatable::batch(:,:)
    integer::i,j
    allocate(batch(last-first+1,x%n_cols));batch=0
    do i=first,last
      do j=x%indptr(i),x%indptr(i+1)-1
        batch(i-first+1,x%indices(j))=batch(i-first+1,x%indices(j))+x%data(j)
      end do
    end do
  end function

  subroutine fit_csr(self,x)
    class(incremental_pca),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::batch(:,:)
    integer::first,last,b
    call check_csr(x)
    call require(self%batch_size>=0,'IncrementalPCA batch_size must be nonnegative')
    b=self%batch_size;if(b==0)b=5*x%n_cols
    self%batch_size_=b;self%n_samples_seen_=0
    first=1
    do while(first<=x%n_rows)
      last=min(x%n_rows,first+b-1)
      if(x%n_rows-last<self%n_components)last=x%n_rows
      batch=dense_batch(x,first,last);call self%partial_fit(batch)
      first=last+1
    end do
  end subroutine

  subroutine partial_fit_csr(self,x)
    class(incremental_pca),intent(inout)::self
    type(csr_matrix),intent(in)::x
    call require(.false.,'IncrementalPCA.partial_fit does not support sparse input; use fit for batched conversion')
  end subroutine

  function transform_csr(self,x)result(z)
    class(incremental_pca),intent(in)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:),batch(:,:)
    integer::first,last
    call require(self%n_samples_seen_>0,'IncrementalPCA is not fitted')
    call check_csr(x)
    call require(x%n_cols==self%n_features_in_,'IncrementalPCA feature count mismatch')
    call require(self%batch_size_>0,'IncrementalPCA sparse transform requires a fitted batch size')
    allocate(z(x%n_rows,self%n_components_));first=1
    do while(first<=x%n_rows)
      last=min(x%n_rows,first+self%batch_size_-1)
      if(x%n_rows-last<self%n_components)last=x%n_rows
      batch=dense_batch(x,first,last);z(first:last,:)=self%transform(batch)
      first=last+1
    end do
  end function

  function fit_transform_csr(self,x)result(z)
    class(incremental_pca),intent(inout)::self
    type(csr_matrix),intent(in)::x
    real(dp),allocatable::z(:,:)
    call self%fit(x);z=self%transform(x)
  end function

  subroutine ipca_fit(self,x)
    class(incremental_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: first,last,n,b
    call check_x(x); n=size(x,1)
    call require(self%batch_size>=0,'IncrementalPCA batch_size must be nonnegative')
    b=self%batch_size
    if(b==0) b=5*size(x,2)
    self%batch_size_=b; self%n_samples_seen_=0
    first=1
    do while(first<=n)
      last=min(n,first+b-1)
      if(n-last<self%n_components) last=n
      call self%partial_fit(x(first:last,:))
      first=last+1
    end do
  end subroutine

  subroutine ipca_partial(self,x)
    class(incremental_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: batch_mean(:),batch_m2(:),delta(:),new_mean(:),new_var(:),a(:,:),s(:),u(:,:),v(:,:)
    real(dp) :: total_variance
    integer :: n,p,k,old,total,m,rank
    call check_x(x); n=size(x,1); p=size(x,2); old=self%n_samples_seen_
    call require(self%n_components>=0.and.self%n_components<=p,'invalid IncrementalPCA n_components')
    k=self%n_components
    if(old==0) then
      call require(n>=2,'IncrementalPCA first batch needs at least two samples')
      if(k==0) k=min(n,p)
      call require(k<=n,'IncrementalPCA first batch is smaller than n_components')
    else
      call require(p==self%n_features_in_,'IncrementalPCA feature count mismatch')
      if(k==0) k=self%n_components_
      call require(k==self%n_components_,'IncrementalPCA component count changed')
    end if
    total=old+n; batch_mean=sum(x,dim=1)/n
    batch_m2=sum((x-spread(batch_mean,1,n))**2,dim=1)
    if(old==0) then
      new_mean=batch_mean; new_var=batch_m2/n; a=x-spread(batch_mean,1,n)
    else
      delta=batch_mean-self%mean_; new_mean=self%mean_+delta*(real(n,dp)/total)
      new_var=(old*self%var_+batch_m2+delta**2*(real(old,dp)/total)*n)/total
      allocate(a(k+n+1,p))
      a(:k,:)=self%components_*spread(self%singular_values_,2,p)
      a(k+1:k+n,:)=x-spread(batch_mean,1,n)
      a(k+n+1,:)=-sqrt((real(old,dp)/total)*n)*delta
    end if
    m=size(a,1); rank=min(m,p); allocate(s(p),u(m,p),v(p,p))
    call dense_svd(a,s,u,v)
    self%components_=transpose(v(:,:k)); self%singular_values_=s(:k)
    self%mean_=new_mean; self%var_=new_var; self%explained_variance_=s(:k)**2/(total-1)
    total_variance=sum(new_var)*total; self%explained_variance_ratio_=s(:k)*0
    if(total_variance>0) self%explained_variance_ratio_=s(:k)**2/total_variance
    self%noise_variance_=0
    if(k/=n.and.k/=p.and.k<rank) self%noise_variance_=sum(s(k+1:rank)**2)/((total-1)*real(rank-k,dp))
    self%n_samples_seen_=total; self%n_features_in_=p; self%n_components_=k
  end subroutine

  function ipca_transform(self,x) result(z)
    class(incremental_pca), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call require(self%n_samples_seen_>0,'IncrementalPCA is not fitted'); call check_x(x)
    call require(size(x,2)==self%n_features_in_,'IncrementalPCA feature count mismatch')
    z=matmul(x-spread(self%mean_,1,size(x,1)),transpose(self%components_))
    if(self%whiten) z=z/spread(max(sqrt(self%explained_variance_),epsilon(1._dp)),1,size(x,1))
  end function

  function ipca_inverse(self,z) result(x)
    class(incremental_pca), intent(in) :: self
    real(dp), intent(in) :: z(:,:)
    real(dp), allocatable :: x(:,:),work(:,:)
    call require(self%n_samples_seen_>0,'IncrementalPCA is not fitted'); call check_x(z)
    call require(size(z,2)==self%n_components_,'IncrementalPCA component width mismatch')
    work=z
    if(self%whiten) work=work*spread(sqrt(self%explained_variance_),1,size(z,1))
    x=matmul(work,self%components_)+spread(self%mean_,1,size(z,1))
  end function

  function ipca_fit_transform(self,x) result(z)
    class(incremental_pca), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%transform(x)
  end function
end module
