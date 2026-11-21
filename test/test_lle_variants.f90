program test_lle_variants
  use sklearn, only: dp,locally_linear_embedding
  use sklearn_common, only: require,symmetric_eigen
  use sklearn_lle_variants, only: lle_hessian_alignment,lle_modified_alignment,lle_local_spectra
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(locally_linear_embedding) :: model,changed
  real(dp) :: x(24,4),original(24,4),eta,objective,identity(2,2),gram(11,11),regularization,residual(11)
  real(dp), allocatable :: z(:,:),again(:,:),matrix(:,:),shifted(:,:),weights(:,:),basis(:,:,:),values(:,:)
  real(dp), allocatable :: spectrum(:),vectors(:,:),local(:,:),query(:,:),other_query(:,:)
  integer, allocatable :: nullity(:)
  integer :: i,j,method
  do i=1,24
    do j=1,4
      x(i,j)=sin(.731_dp*i*j)+cos(.213_dp*i*i+.191_dp*j)+.02_dp*i*j
    end do
  end do
  original=x; model%n_neighbors=11; model%n_components=2
  identity=0; identity(1,1)=1; identity(2,2)=1
  do method=1,2
    model%method='hessian'
    if(method==2) model%method='modified'
    z=model%fit_transform(x)
    call require(all(x==original),'new LLE methods preserve input')
    if(method==1) then
      matrix=lle_hessian_alignment(x,model%neighbors_,2,model%hessian_tol)
      shifted=lle_hessian_alignment(3*x+10,model%neighbors_,2,model%hessian_tol)
    else
      matrix=lle_modified_alignment(x,model%neighbors_,2,model%modified_tol,nullity,weights,eta)
      shifted=lle_modified_alignment(3*x+10,model%neighbors_,2,model%modified_tol)
      call require(all(nullity>=7).and.all(nullity<11),'adaptive almost-null-space dimensions')
      call require(eta>=0.and.ieee_is_finite(eta),'median spectral ratio')
      call require(maxval(abs(sum(weights,dim=2)-1))<1.e-11_dp,'modified affine regularized weights')
      call lle_local_spectra(x,model%neighbors_,basis,values)
      do i=1,24
        local=x(model%neighbors_(i,:),:)-spread(x(i,:),1,11); gram=matmul(local,transpose(local))
        regularization=.001_dp*sum(values(i,:))
        do j=1,11
          gram(j,j)=gram(j,j)+regularization
        end do
        residual=matmul(gram,weights(i,:)); residual=residual-residual(1)
        call require(maxval(abs(residual))<1.e-9_dp,'regularized weights solve constrained local covariance')
      end do
    end if
    call require(maxval(abs(matrix-transpose(matrix)))<1.e-11_dp,'symmetric LLE alignment')
    call require(maxval(abs(sum(matrix,dim=2)))<1.e-9_dp,'alignment annihilates constant mode')
    call require(maxval(abs(matrix-shifted))<1.e-8_dp,'alignment translation and scale invariance')
    allocate(spectrum(24),vectors(24,24)); call symmetric_eigen(matrix,spectrum,vectors)
    call require(minval(spectrum)>-1.e-10_dp,'positive semidefinite local alignment sum')
    deallocate(spectrum,vectors)
    call require(maxval(abs(matmul(transpose(z),z)-identity))<1.e-11_dp,'embedding orthonormal columns')
    call require(maxval(abs(sum(z,dim=1)))<1.e-9_dp,'embedding removes constant mode')
    objective=sum(z*matmul(matrix,z))
    call require(abs(objective-model%reconstruction_error_)<1.e-9_dp,'reported alignment energy')
    again=model%fit_transform(x); call require(all(z==again),'deterministic variant refit')
    query=model%transform(x(:5,:)+.017_dp)
    call require(all(shape(query)==[5,2]).and.all(ieee_is_finite(query)),'barycentric query transform')
    changed=model; changed%reg=.1_dp; again=changed%fit_transform(x)
    call require(all(z==again),'nonstandard LLE fit ignores barycentric reg as in source')
    other_query=changed%transform(x(:5,:)+.017_dp)
    call require(maxval(abs(query-other_query))>1.e-9_dp,'query transform honors barycentric regularization')
  end do
  print *, 'Hessian and modified LLE: PASS'
end program
