! sklearn 1.6.1 metrics/_classification.py probability-loss contracts (BSD).
! NumPy pairwise reductions follow retained NumPy 2.2.6 BSD terms.
module sklearn_probability_metrics
  use sklearn_common, only: dp,require,unique_labels,label_index
  use sklearn_label_binarization, only: label_binarizer
  use sklearn_sparse, only: csr_matrix
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  implicit none
  private
  public :: log_loss,d2_log_loss_score
  ! Typed public adapters are declared below the shared implementation.
  interface log_loss
    module procedure loss_iv,loss_im,loss_bv,loss_bm,loss_mv,loss_mm,loss_lv,loss_lm,loss_sv,loss_sm
  end interface
  interface d2_log_loss_score
    module procedure d2_iv,d2_im,d2_bv,d2_bm,d2_mv,d2_mm,d2_lv,d2_lm,d2_sv,d2_sm
  end interface
contains
  recursive real(dp) function numpy_sum(a) result(total)
    real(dp), intent(in) :: a(:)
    real(dp) :: partial(8),left,right
    integer :: n,i,j,whole,split
    n=size(a);total=-0._dp
    if(n<8) then
      do i=1,n
        total=ieee_difference(total,-a(i))
      end do
    else if(n<=128) then
      partial=a(:8);whole=n-mod(n,8)
      do i=9,whole,8
        do j=1,8
          partial(j)=ieee_difference(partial(j),-a(i+j-1))
        end do
      end do
      left=ieee_difference(ieee_difference(partial(1),-partial(2)),-ieee_difference(partial(3),-partial(4)))
      right=ieee_difference(ieee_difference(partial(5),-partial(6)),-ieee_difference(partial(7),-partial(8)))
      total=ieee_difference(left,-right)
      do i=whole+1,n
        total=ieee_difference(total,-a(i))
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=ieee_difference(numpy_sum(a(:split)),-numpy_sum(a(split+1:)))
    end if
    ! NumPy's outer add reduction includes its positive-zero identity.
    if(.not.ieee_is_nan(total)) then
      if(total==0) total=0._dp
    end if
  end function
  subroutine check_probabilities(p,n,sample_weight)
    real(dp), intent(in) :: p(:,:)
    integer, intent(in) :: n
    real(dp), optional, intent(in) :: sample_weight(:)
    call require(size(p,1)>0.and.size(p,2)>0,'probability metrics need nonempty predictions')
    call require(size(p,1)==n,'probability metric sample counts differ')
    call require(all(ieee_is_finite(p)),'probability predictions must be finite')
    if(present(sample_weight)) call require(size(sample_weight)==n,'probability metric sample weight length mismatch')
  end subroutine
  function vector_target(y,labels) result(target)
    integer, intent(in) :: y(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), allocatable :: target(:,:)
    integer, allocatable :: codes(:,:)
    type(label_binarizer) :: binarizer
    if(present(labels)) then
      call binarizer%fit(labels)
    else
      call binarizer%fit(y)
    end if
    call require(size(binarizer%classes_)>=2,'log_loss needs at least two fitted classes')
    codes=binarizer%transform(y)
    if(size(codes,2)==1) then
      allocate(target(size(y),2));target(:,2)=real(codes(:,1),dp);target(:,1)=1-target(:,2)
    else
      target=real(codes,dp)
    end if
  end function
  function matrix_target(y,labels) result(target)
    integer, intent(in) :: y(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), allocatable :: target(:,:)
    integer :: first,second,i,j
    logical :: seen_second
    call require(size(y,1)>0.and.size(y,2)>0,'log_loss target needs nonempty axes')
    if(size(y,2)==1) then
      target=vector_target(y(:,1),labels);return
    end if
    call require(.not.present(labels),'log_loss vector labels cannot fit indicator targets')
    first=y(1,1);second=first;seen_second=.false.
    do j=1,size(y,2)
      do i=1,size(y,1)
        if(y(i,j)==first) cycle
        if(seen_second) then
          call require(y(i,j)==second,'log_loss target is multiclass multioutput rather than an indicator')
        else
          second=y(i,j);seen_second=.true.
        end if
      end do
    end do
    target=real(y,dp)
  end function
  function sparse_target(y,labels) result(target)
    type(csr_matrix), intent(in) :: y
    integer, optional, intent(in) :: labels(:)
    real(dp), allocatable :: target(:,:)
    call require(.false.,'log_loss baseline rejects sparse targets')
  end function
  real(dp) function loss_core(target,p,sample_weight,normalize) result(value)
    real(dp), intent(in) :: target(:,:),p(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    real(dp), allocatable :: probabilities(:,:),losses(:),products(:),terms(:)
    real(dp) :: denominator
    integer :: n,q,i,j
    logical :: norm
    n=size(target,1);q=size(target,2);call check_probabilities(p,n,sample_weight)
    if(size(p,2)==1) then
      allocate(probabilities(n,2));probabilities(:,2)=p(:,1)
      do i=1,n
        probabilities(i,1)=ieee_difference(1._dp,p(i,1))
      end do
    else
      probabilities=p
    end if
    call require(size(probabilities,2)==q,'log_loss probability class count mismatch')
    probabilities=max(epsilon(1._dp),min(1-epsilon(1._dp),probabilities))
    allocate(losses(n),terms(q),products(n))
    do i=1,n
      terms=0
      do j=1,q
        if(target(i,j)/=0) terms(j)=ieee_product(target(i,j),log(probabilities(i,j)))
      end do
      losses(i)=-numpy_sum(terms)
    end do
    norm=.true.;if(present(normalize)) norm=normalize
    if(present(sample_weight)) then
      do i=1,n
        products(i)=ieee_product(losses(i),sample_weight(i))
      end do
      if(norm) then
        denominator=numpy_sum(sample_weight)
        if(.not.ieee_is_nan(denominator)) call require(denominator/=0,'log_loss averaging weights have zero total')
        value=ieee_quotient(numpy_sum(products),denominator)
      else
        ! The source uses NumPy dot rather than a pairwise sum here.
        value=0
        do i=1,n
          value=ieee_difference(value,-products(i))
        end do
      end if
    else
      value=numpy_sum(losses)
      if(norm) value=ieee_quotient(value,real(n,dp))
    end if
  end function
  real(dp) function d2_vector(y,p,labels,sample_weight) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: p(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: target(:,:),weights(:),counts(:),null_prob(:,:)
    integer, allocatable :: classes(:)
    real(dp) :: numerator,denominator,total
    integer :: i,j
    call check_probabilities(p,size(y),sample_weight)
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(size(y)<2) return
    target=vector_target(y,labels);numerator=loss_core(target,p,sample_weight,.false.)
    allocate(weights(size(y)));weights=1
    if(present(sample_weight)) weights=sample_weight
    call require(all(ieee_is_finite(weights)),'D2 log loss sample weights must be finite')
    classes=unique_labels(y);allocate(counts(size(classes)));counts=0
    do i=1,size(y)
      j=label_index(classes,y(i));counts(j)=ieee_difference(counts(j),-weights(i))
    end do
    total=numpy_sum(weights)
    do j=1,size(counts)
      counts(j)=ieee_quotient(counts(j),total)
    end do
    allocate(null_prob(size(y),size(counts)))
    do i=1,size(y)
      null_prob(i,:)=counts
    end do
    denominator=loss_core(target,null_prob,sample_weight,.false.)
    if(.not.ieee_is_nan(denominator)) call require(denominator/=0,'D2 log loss null-model loss is zero')
    value=ieee_difference(1._dp,ieee_quotient(numerator,denominator))
  end function
  real(dp) function d2_matrix(y,p,labels,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: p(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    real(dp), allocatable :: target(:,:)
    call check_probabilities(p,size(y,1),sample_weight)
    value=ieee_value(0._dp,ieee_quiet_nan)
    if(size(y,1)<2) return
    target=matrix_target(y,labels);value=loss_core(target,p,sample_weight,.false.)
    if(present(sample_weight)) call require(all(ieee_is_finite(sample_weight)),'D2 log loss sample weights must be finite')
    call require(.false.,'D2 log loss installed NumPy baseline rejects matrix targets in bincount')
  end function
  real(dp) function d2_sparse(y,p,labels,sample_weight) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: p(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=0
    call require(.false.,'D2 log loss baseline rejects sparse targets')
  end function
  real(dp) function loss_iv(y,prob,labels,sample_weight,normalize) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(vector_target(y,labels),reshape(prob,[size(prob),1]),sample_weight,normalize)
  end function
  real(dp) function d2_iv(y,prob,labels,sample_weight) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_vector(y,reshape(prob,[size(prob),1]),labels,sample_weight)
  end function
  real(dp) function loss_im(y,prob,labels,sample_weight,normalize) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(vector_target(y,labels),prob,sample_weight,normalize)
  end function
  real(dp) function d2_im(y,prob,labels,sample_weight) result(value)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_vector(y,prob,labels,sample_weight)
  end function
  real(dp) function loss_bv(y,prob,labels,sample_weight,normalize) result(value)
    logical, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(vector_target(merge(1,0,y),labels),reshape(prob,[size(prob),1]),sample_weight,normalize)
  end function
  real(dp) function d2_bv(y,prob,labels,sample_weight) result(value)
    logical, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_vector(merge(1,0,y),reshape(prob,[size(prob),1]),labels,sample_weight)
  end function
  real(dp) function loss_bm(y,prob,labels,sample_weight,normalize) result(value)
    logical, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(vector_target(merge(1,0,y),labels),prob,sample_weight,normalize)
  end function
  real(dp) function d2_bm(y,prob,labels,sample_weight) result(value)
    logical, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_vector(merge(1,0,y),prob,labels,sample_weight)
  end function
  real(dp) function loss_mv(y,prob,labels,sample_weight,normalize) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(matrix_target(y,labels),reshape(prob,[size(prob),1]),sample_weight,normalize)
  end function
  real(dp) function d2_mv(y,prob,labels,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_matrix(y,reshape(prob,[size(prob),1]),labels,sample_weight)
  end function
  real(dp) function loss_mm(y,prob,labels,sample_weight,normalize) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(matrix_target(y,labels),prob,sample_weight,normalize)
  end function
  real(dp) function d2_mm(y,prob,labels,sample_weight) result(value)
    integer, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_matrix(y,prob,labels,sample_weight)
  end function
  real(dp) function loss_lv(y,prob,labels,sample_weight,normalize) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(matrix_target(merge(1,0,y),labels),reshape(prob,[size(prob),1]),sample_weight,normalize)
  end function
  real(dp) function d2_lv(y,prob,labels,sample_weight) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_matrix(merge(1,0,y),reshape(prob,[size(prob),1]),labels,sample_weight)
  end function
  real(dp) function loss_lm(y,prob,labels,sample_weight,normalize) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(matrix_target(merge(1,0,y),labels),prob,sample_weight,normalize)
  end function
  real(dp) function d2_lm(y,prob,labels,sample_weight) result(value)
    logical, intent(in) :: y(:,:)
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_matrix(merge(1,0,y),prob,labels,sample_weight)
  end function
  real(dp) function loss_sv(y,prob,labels,sample_weight,normalize) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(sparse_target(y,labels),reshape(prob,[size(prob),1]),sample_weight,normalize)
  end function
  real(dp) function d2_sv(y,prob,labels,sample_weight) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: prob(:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_sparse(y,reshape(prob,[size(prob),1]),labels,sample_weight)
  end function
  real(dp) function loss_sm(y,prob,labels,sample_weight,normalize) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    logical, optional, intent(in) :: normalize
    value=loss_core(sparse_target(y,labels),prob,sample_weight,normalize)
  end function
  real(dp) function d2_sm(y,prob,labels,sample_weight) result(value)
    type(csr_matrix), intent(in) :: y
    real(dp), intent(in) :: prob(:,:)
    integer, optional, intent(in) :: labels(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    value=d2_sparse(y,prob,labels,sample_weight)
  end function
end module
