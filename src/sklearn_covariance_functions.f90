! sklearn 1.6.1 covariance functional APIs (BSD-3-Clause).
! Typed result/storage adapters and IEEE composition are original MIT.
module sklearn_covariance_functions
  use sklearn_common,only:dp,require,check_x
  use sklearn_shrinkage,only:ledoit_wolf_shrinkage
  use sklearn_kernel_dispatch,only:kernel_blas_inner_product
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  private
  public::covariance_estimate,empirical_array,empirical_vector,ledoit_array,oas_array
  public::shrink_array_2, &
    shrink_array_3, &
    shrink_array_4, &
    shrink_array_5, &
    shrink_array_6, &
    shrink_array_7, &
    shrink_array_8, &
    shrink_array_9, &
    shrink_array_10, &
    shrink_array_11, &
    shrink_array_12, &
    shrink_array_13, &
    shrink_array_14, &
    shrink_array_15
  type::covariance_estimate
    real(dp),allocatable::covariance(:,:)
    real(dp)::shrinkage=0
  end type
contains
  subroutine quiet(modes,flags)
    logical,intent(out)::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_get_flag(ieee_all,flags);call ieee_get_halting_mode(ieee_all,modes)
    call ieee_set_halting_mode(ieee_all,.false.)
  end subroutine
  subroutine restore(modes,flags)
    logical,intent(in)::modes(size(ieee_all)),flags(size(ieee_all))
    call ieee_set_halting_mode(ieee_all,modes);call ieee_set_flag(ieee_all,flags)
  end subroutine
  recursive real(dp) function numpy_sum(values) result(total)
    ! NumPy 2.2.6 loops_utils.h.src pairwise reduction; retained NumPy BSD notice.
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,split,whole
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8);whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2;split=split-mod(split,8)
      total=numpy_sum(values(:split))+numpy_sum(values(split+1:))
    end if
  end function
  function empirical_vector(x,assume_centered)result(covariance)
    real(dp),intent(in)::x(:)
    logical,optional,intent(in)::assume_centered
    real(dp),allocatable::covariance(:,:)
    covariance=empirical_array(reshape(x,[1,size(x)]),assume_centered)
  end function
  function empirical_array(x,assume_centered)result(covariance)
    real(dp),intent(in)::x(:,:)
    logical,optional,intent(in)::assume_centered
    real(dp),allocatable::covariance(:,:),centered(:,:),location(:)
    real(dp)::value
    integer::n,p,i,j,k
    logical::center,modes(size(ieee_all)),flags(size(ieee_all))
    n=size(x,1);p=size(x,2)
    call require(n>0.and.p>0,'empirical_covariance requires nonempty samples and features')
    center=.true.;if(present(assume_centered))center=.not.assume_centered
    call quiet(modes,flags)
    centered=x;allocate(location(p),covariance(p,p));location=0
    if(center)then
      if(p==1)then
        location(1)=numpy_sum(x(:,1))/n
      else
        do i=1,n
          location=location+x(i,:)
        end do
        location=location/n
      end if
      centered=centered-spread(location,1,n)
    end if
    do j=1,p
      do i=1,p
        if(p==1)then
          value=kernel_blas_inner_product(centered(:,i),centered(:,j))
        else
          value=0
          do k=1,n
            value=ieee_fma(centered(k,i),centered(k,j),value)
          end do
        end if
        if(center)then
          covariance(i,j)=value*(1._dp/n)
        else
          covariance(i,j)=value/n
        end if
      end do
    end do
    call restore(modes,flags)
  end function
  function shrink_batch(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:),diagonal(:)
    real(dp)::amount,mu,eye
    integer::batch,i,j,p
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    amount=.1_dp;if(present(shrinkage))amount=shrinkage
    call require(ieee_is_finite(amount),'covariance shrinkage must be finite')
    call require(amount>=0.and.amount<=1,'covariance shrinkage must be between zero and one')
    p=size(a,3)
    call require(size(a,1)>0.and.p>0.and.size(a,2)==p,'shrunk_covariance requires nonempty square matrices')
    call require(all(ieee_is_finite(a)),'shrunk_covariance input must be finite')
    allocate(out(size(a,1),p,p),diagonal(p));call quiet(modes,flags)
    do batch=1,size(a,1)
      do i=1,p
        diagonal(i)=a(batch,i,i)
      end do
      mu=numpy_sum(diagonal)/p
      do j=1,p
        do i=1,p
          eye=0;if(i==j)eye=1
          out(batch,i,j)=(1-amount)*a(batch,i,j)+(amount*mu)*eye
        end do
      end do
    end do
    call restore(modes,flags)
  end function
  function ledoit_array(x,assume_centered,block_size)result(estimate)
    real(dp),intent(in)::x(:,:)
    logical,optional,intent(in)::assume_centered
    integer,optional,intent(in)::block_size
    type(covariance_estimate)::estimate
    real(dp),allocatable::diagonal(:)
    real(dp)::mu
    logical::modes(size(ieee_all)),flags(size(ieee_all))
    integer::p,j,block
    call check_x(x);block=1000;if(present(block_size))block=block_size
    call require(block>=1,'Ledoit-Wolf block_size must be positive')
    if(size(x,2)==1)then
      estimate=oas_array(x,assume_centered);return
    end if
    estimate%shrinkage=ledoit_wolf_shrinkage(x,assume_centered,block)
    estimate%covariance=empirical_array(x,assume_centered);p=size(x,2)
    allocate(diagonal(p));call quiet(modes,flags)
    do j=1,p
      diagonal(j)=estimate%covariance(j,j)
    end do
    mu=numpy_sum(diagonal)/p
    estimate%covariance=(1-estimate%shrinkage)*estimate%covariance
    do j=1,p
      estimate%covariance(j,j)=estimate%covariance(j,j)+estimate%shrinkage*mu
    end do
    call require(all(ieee_is_finite(estimate%covariance)),'Ledoit-Wolf covariance exceeds finite working range')
    call restore(modes,flags)
  end function
  function oas_array(x,assume_centered)result(estimate)
    real(dp),intent(in)::x(:,:)
    logical,optional,intent(in)::assume_centered
    type(covariance_estimate)::estimate
    real(dp),allocatable::diagonal(:),squared(:),centered(:)
    real(dp)::mu,alpha,den,num
    logical::center,modes(size(ieee_all)),flags(size(ieee_all))
    integer::p,j
    call check_x(x);p=size(x,2);center=.true.
    if(present(assume_centered))center=.not.assume_centered
    call quiet(modes,flags)
    if(p==1)then
      centered=x(:,1)
      if(center)centered=centered-numpy_sum(centered)/size(centered)
      allocate(estimate%covariance(1,1));estimate%covariance(1,1)=numpy_sum(centered**2)/size(centered)
      estimate%shrinkage=0
    else
      estimate%covariance=empirical_array(x,assume_centered);allocate(diagonal(p))
      do j=1,p
        diagonal(j)=estimate%covariance(j,j)
      end do
      mu=numpy_sum(diagonal)/p;squared=reshape(estimate%covariance**2,[p*p]);alpha=numpy_sum(squared)/(p*p)
      num=alpha+mu**2;den=(size(x,1)+1)*(alpha-mu**2/p)
      estimate%shrinkage=1
      if(den/=0)estimate%shrinkage=min(num/den,1._dp)
      estimate%covariance=(1-estimate%shrinkage)*estimate%covariance
      do j=1,p
        estimate%covariance(j,j)=estimate%covariance(j,j)+estimate%shrinkage*mu
      end do
    end if
    call require(all(ieee_is_finite(estimate%covariance)),'OAS covariance exceeds finite working range')
    call restore(modes,flags)
  end function
  function shrink_array_2(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:),b(:,:,:),c(:,:,:)
    integer::extents(2),batches
    extents=[size(a,1),size(a,2)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,1)*size(a,2))
    b=reshape(a,[batches,size(a,1),size(a,2)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_3(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(3),batches
    extents=[size(a,1),size(a,2),size(a,3)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,2)*size(a,3))
    b=reshape(a,[batches,size(a,2),size(a,3)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_4(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(4),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,3)*size(a,4))
    b=reshape(a,[batches,size(a,3),size(a,4)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_5(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(5),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,4)*size(a,5))
    b=reshape(a,[batches,size(a,4),size(a,5)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_6(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(6),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,5)*size(a,6))
    b=reshape(a,[batches,size(a,5),size(a,6)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_7(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(7),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,6)*size(a,7))
    b=reshape(a,[batches,size(a,6),size(a,7)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_8(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(8),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,7)*size(a,8))
    b=reshape(a,[batches,size(a,7),size(a,8)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_9(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(9),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,8)*size(a,9))
    b=reshape(a,[batches,size(a,8),size(a,9)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_10(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(10),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9),size(a,10)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,9)*size(a,10))
    b=reshape(a,[batches,size(a,9),size(a,10)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_11(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(11),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9),size(a,10), &
      size(a,11)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,10)*size(a,11))
    b=reshape(a,[batches,size(a,10),size(a,11)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_12(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(12),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9),size(a,10), &
      size(a,11),size(a,12)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,11)*size(a,12))
    b=reshape(a,[batches,size(a,11),size(a,12)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_13(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(13),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9),size(a,10), &
      size(a,11),size(a,12),size(a,13)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,12)*size(a,13))
    b=reshape(a,[batches,size(a,12),size(a,13)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_14(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(14),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9),size(a,10), &
      size(a,11),size(a,12),size(a,13),size(a,14)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,13)*size(a,14))
    b=reshape(a,[batches,size(a,13),size(a,14)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
  function shrink_array_15(a,shrinkage)result(out)
    real(dp),intent(in)::a(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:)
    real(dp),optional,intent(in)::shrinkage
    real(dp),allocatable::out(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:),b(:,:,:),c(:,:,:)
    integer::extents(15),batches
    extents=[size(a,1),size(a,2),size(a,3),size(a,4),size(a,5), &
      size(a,6),size(a,7),size(a,8),size(a,9),size(a,10), &
      size(a,11),size(a,12),size(a,13),size(a,14),size(a,15)]
    call require(all(extents>0),'shrunk_covariance requires positive array extents')
    batches=size(a)/(size(a,14)*size(a,15))
    b=reshape(a,[batches,size(a,14),size(a,15)]);c=shrink_batch(b,shrinkage)
    out=reshape(c,extents)
  end function
end module
