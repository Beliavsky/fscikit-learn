! sklearn 1.6.1 decomposition/_nmf.py sparse storage contracts (BSD-3-Clause).
! Owned CSR canonicalization and products are original MIT integration.
module sklearn_nmf_sparse_data
  use sklearn_common,only:dp,require
  use sklearn_sparse,only:csr_matrix
  use sklearn_stateless_preprocessing,only:sorted_sparse
  use,intrinsic::iso_fortran_env,only:int64
  use,intrinsic::ieee_arithmetic,only:ieee_is_finite
  implicit none
  private
  public::nmf_prepare_sparse,nmf_sparse_cross
contains
  function nmf_prepare_sparse(x,beta,solver)result(z)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::beta
    character(*),intent(in)::solver
    type(csr_matrix)::z
    call x%validate()
    call require(x%n_rows>0.and.x%n_cols>0,'NMF CSR shape must be nonempty')
    call require(all(ieee_is_finite(x%data)),'NMF CSR data must be finite')
    ! Source X.min() calls SciPy sum_duplicates before initialization/updates.
    ! Preserve the caller's arrays, including explicit zeros, in an owned copy.
    z=sorted_sparse(x,.true.)
    call require(all(ieee_is_finite(z%data)),'NMF canonical CSR data must be finite')
    call require(all(z%data>=0),'NMF requires nonnegative X')
    call require(all(z%data<=huge(1._dp)**(1._dp/3)/(64*real(max(x%n_rows,x%n_cols),dp))), &
      'NMF data exceeds working range')
    if(beta<=0)then
      call require(int(size(z%data),int64)==int(z%n_rows,int64)*z%n_cols.and.all(z%data>0), &
        'NMF nonpositive beta requires strictly positive X')
    end if
    ! _special_sparse_dot drops stored zeros, so Source's data arrays no longer
    ! align for non-Frobenius MU losses. Retain that actual Source rejection.
    if(solver=='mu'.and.beta/=2)call require(all(z%data/=0),'NMF sparse beta loss rejects explicit zeros')
  end function
  function nmf_sparse_cross(x,right,transposed)result(value)
    type(csr_matrix),intent(in)::x
    real(dp),intent(in)::right(:,:)
    logical,intent(in)::transposed
    real(dp),allocatable::value(:,:)
    integer::i,j,e
    if(.not.transposed)then
      value=x%matmul(right)
    else
      allocate(value(x%n_cols,size(right,2)));value=0
      do i=1,x%n_rows
        do e=x%indptr(i),x%indptr(i+1)-1
          j=x%indices(e);value(j,:)=value(j,:)+x%data(e)*right(i,:)
        end do
      end do
    end if
  end function
end module
