! Dense standard/Hessian/modified LLE and LTSA adapted from sklearn/manifold/_locally_linear.py.
! See THIRD_PARTY_NOTICES.md for upstream BSD-3-Clause attribution.
module sklearn_locally_linear
  use sklearn_common
  use sklearn_lle_variants, only: lle_hessian_alignment,lle_modified_alignment
  use sklearn_metrics, only: pairwise_distances
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: locally_linear_embedding
  type :: locally_linear_embedding
    integer :: n_neighbors=5,n_components=2
    real(dp) :: reg=1.e-3_dp,reconstruction_error_=0,hessian_tol=1.e-4_dp,modified_tol=1.e-12_dp
    character(8) :: method='standard'
    real(dp), allocatable :: embedding_(:,:),x_fit_(:,:),weights_(:,:)
    integer, allocatable :: neighbors_(:,:)
  contains
    procedure :: fit=>lle_fit
    procedure :: fit_transform=>lle_fit_transform
    procedure :: transform=>lle_transform
  end type
contains
  function barycenter(point,neighbors,reg) result(weights)
    real(dp), intent(in) :: point(:),neighbors(:,:),reg
    real(dp) :: weights(size(neighbors,1)),local(size(neighbors,1),size(neighbors,2))
    real(dp) :: gram(size(neighbors,1),size(neighbors,1)),trace,penalty,denom
    integer :: j,k
    k=size(neighbors,1); local=neighbors-spread(point,1,k); gram=matmul(local,transpose(local)); trace=0
    do j=1,k
      trace=trace+gram(j,j)
    end do
    penalty=reg
    if(trace>0) penalty=reg*trace
    do j=1,k
      gram(j,j)=gram(j,j)+penalty
    end do
    weights=solve_system(gram,spread(1._dp,1,k)); denom=sum(weights)
    call require(abs(denom)>tiny(1._dp),'LLE zero barycentric weight sum')
    weights=weights/denom
  end function
  subroutine lle_fit(self,x)
    class(locally_linear_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: distances(:,:),w(:,:),alignment(:,:),values(:),vectors(:,:),basis(:,:)
    real(dp), allocatable :: local(:,:),local_values(:),local_vectors(:,:),projector(:,:)
    integer, allocatable :: order(:)
    integer :: n,p,k,d,i,j,a,b,column
    call check_x(x); n=size(x,1); p=size(x,2); k=self%n_neighbors; d=self%n_components
    call require(k>0.and.k<n,'LLE requires 0 < n_neighbors < n_samples')
    call require(d>0.and.d<=p.and.d<n,'invalid LLE n_components')
    call require(ieee_is_finite(self%reg),'invalid LLE regularization')
    call require(self%reg>=0,'invalid LLE regularization')
    call require(ieee_is_finite(self%hessian_tol),'LLE hessian_tol must be finite')
    call require(self%hessian_tol>=0,'LLE hessian_tol must be nonnegative')
    call require(ieee_is_finite(self%modified_tol),'LLE modified_tol must be finite')
    call require(self%modified_tol>=0,'LLE modified_tol must be nonnegative')
    call require(self%method=='standard'.or.self%method=='ltsa'.or.self%method=='hessian'.or. &
      self%method=='modified','unsupported LLE method')
    if(self%method=='ltsa') call require(k>d,'LTSA needs n_neighbors > n_components')
    self%x_fit_=x; distances=pairwise_distances(x,x)
    if(allocated(self%neighbors_)) deallocate(self%neighbors_,self%weights_)
    allocate(self%neighbors_(n,k),self%weights_(n,k),w(n,n),alignment(n,n)); w=0; alignment=0
    self%weights_=0
    do i=1,n
      distances(i,i)=huge(1._dp); order=argsort(distances(i,:)); self%neighbors_(i,:)=order(:k)
      if(self%method=='standard') then
        self%weights_(i,:)=barycenter(x(i,:),x(order(:k),:),self%reg)
        w(i,order(:k))=self%weights_(i,:)
      end if
    end do
    if(self%method=='standard') then
      w=-w
      do i=1,n
        w(i,i)=w(i,i)+1
      end do
      alignment=matmul(transpose(w),w)
    else if(self%method=='hessian') then
      alignment=lle_hessian_alignment(x,self%neighbors_,d,self%hessian_tol)
    else if(self%method=='modified') then
      alignment=lle_modified_alignment(x,self%neighbors_,d,self%modified_tol)
    else
      allocate(local_values(k),local_vectors(k,k))
      do i=1,n
        order=self%neighbors_(i,:); local=x(order,:)
        local=local-spread(sum(local,dim=1)/k,1,k)
        call symmetric_eigen(matmul(local,transpose(local)),local_values,local_vectors)
        projector=matmul(local_vectors(:,:d),transpose(local_vectors(:,:d)))+1._dp/k
        do a=1,k
          do b=1,k
            alignment(order(a),order(b))=alignment(order(a),order(b))-projector(a,b)
          end do
          alignment(order(a),order(a))=alignment(order(a),order(a))+1
        end do
      end do
    end if
    ! The constant vector is an exact null mode. Remove it analytically rather
    ! than identifying it among numerically almost-zero alignment eigenvalues.
    ! Helmert columns are an orthonormal basis perpendicular to that vector.
    allocate(basis(n,n-1)); basis=0
    do j=1,n-1
      basis(:j,j)=1/sqrt(real(j,dp)*(j+1))
      basis(j+1,j)=-j/sqrt(real(j,dp)*(j+1))
    end do
    allocate(values(n-1),vectors(n-1,n-1))
    call symmetric_eigen(matmul(transpose(basis),matmul(alignment,basis)),values,vectors)
    if(allocated(self%embedding_)) deallocate(self%embedding_)
    allocate(self%embedding_(n,d)); self%reconstruction_error_=0
    do j=1,d
      column=n-j; self%embedding_(:,j)=matmul(basis,vectors(:,column))
      self%reconstruction_error_=self%reconstruction_error_+values(column)
      i=maxloc(abs(self%embedding_(:,j)),dim=1)
      if(self%embedding_(i,j)<0) self%embedding_(:,j)=-self%embedding_(:,j)
    end do
    self%reconstruction_error_=max(0._dp,self%reconstruction_error_)
  end subroutine
  function lle_transform(self,x) result(z)
    class(locally_linear_embedding), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:),distances(:,:),weights(:)
    integer, allocatable :: order(:)
    integer :: i,k
    call require(allocated(self%embedding_),'LLE is not fitted')
    call check_xy(transpose(x),size(self%x_fit_,2)); k=size(self%neighbors_,2)
    distances=pairwise_distances(x,self%x_fit_); allocate(z(size(x,1),size(self%embedding_,2)))
    do i=1,size(x,1)
      order=argsort(distances(i,:)); weights=barycenter(x(i,:),self%x_fit_(order(:k),:),self%reg)
      z(i,:)=matmul(weights,self%embedding_(order(:k),:))
    end do
  end function
  function lle_fit_transform(self,x) result(z)
    class(locally_linear_embedding), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: z(:,:)
    call self%fit(x); z=self%embedding_
  end function
end module
