! LIBSVM stored-node kernels and owned CSR sorting/gathers.
! sklearn 1.6.1 / LIBSVM BSD notices retained in LICENSES/.
module sklearn_libsvm_sparse
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix
  implicit none
  private
  public::sparse_inner,svm_sorted,sparse_rows,libsvm_sparse_gram
contains
  real(dp) function sparse_inner(x,i,y,j,distance) result(value)
    type(csr_matrix),intent(in)::x,y
    integer,intent(in)::i,j
    logical,intent(in)::distance
    integer::a,b,ae,be
    value=0;a=x%indptr(i);ae=x%indptr(i+1);b=y%indptr(j);be=y%indptr(j+1)
    do while(a<ae.and.b<be)
      if(x%indices(a)==y%indices(b))then
        if(distance)then
          value=value+(x%data(a)-y%data(b))**2
        else
          value=value+x%data(a)*y%data(b)
        end if
        a=a+1;b=b+1
      else if(x%indices(a)<y%indices(b))then
        if(distance)value=value+x%data(a)**2
        a=a+1
      else
        if(distance)value=value+y%data(b)**2
        b=b+1
      end if
    end do
    if(distance)then
      do while(a<ae)
        value=value+x%data(a)**2;a=a+1
      end do
      do while(b<be)
        value=value+y%data(b)**2;b=b+1
      end do
    end if
  end function
  function svm_sorted(x) result(y)
    type(csr_matrix),intent(in)::x
    type(csr_matrix)::y
    integer::i,a,b,j,k,key,depth,length
    real(dp)::value
    logical::ordered
    y=x;ordered=.true.
    do i=1,x%n_rows
      a=x%indptr(i);b=x%indptr(i+1)-1
      if(any(x%indices(a+1:b)<x%indices(a:b-1)))ordered=.false.
    end do
    if(ordered)return
    ! SciPy csr_sort_indices sorts every row with std::sort when any row is
    ! unordered. Preserve the reference wheel's median/partition tie order.
    do i=1,x%n_rows
      a=x%indptr(i);b=x%indptr(i+1);length=b-a;depth=0
      do while(length>1)
        length=length/2;depth=depth+2
      end do
      call partition_sort(a,b,depth)
      do j=a+1,b-1
        key=y%indices(j);value=y%data(j);k=j
        do while(k>a)
          if(y%indices(k-1)<=key)exit
          y%indices(k)=y%indices(k-1);y%data(k)=y%data(k-1);k=k-1
        end do
        y%indices(k)=key;y%data(k)=value
      end do
    end do
  contains
    subroutine swap(a,b)
      integer,intent(in)::a,b
      integer::key
      real(dp)::value
      key=y%indices(a);y%indices(a)=y%indices(b);y%indices(b)=key
      value=y%data(a);y%data(a)=y%data(b);y%data(b)=value
    end subroutine
    recursive subroutine partition_sort(lo,hi,depth)
      integer,intent(in)::lo,hi,depth
      integer::a,b,c,p,left,right,start,last
      if(hi-lo<=16)return
      if(depth==0)then
        ! Bounded-memory heap fallback for adversarial partition sequences.
        do start=(hi-lo)/2,1,-1
          call sift(start,hi-lo,lo)
        end do
        do last=hi-lo,2,-1
          call swap(lo,lo+last-1);call sift(1,last-1,lo)
        end do
        return
      end if
      a=lo+1;b=lo+(hi-lo)/2;c=hi-1
      if(y%indices(a)<y%indices(b))then
        if(y%indices(b)<y%indices(c))then
          p=b
        else if(y%indices(a)<y%indices(c))then
          p=c
        else
          p=a
        end if
      else
        if(y%indices(a)<y%indices(c))then
          p=a
        else if(y%indices(b)<y%indices(c))then
          p=c
        else
          p=b
        end if
      end if
      call swap(lo,p);left=lo+1;right=hi
      do
        do while(y%indices(left)<y%indices(lo))
          left=left+1
        end do
        right=right-1
        do while(y%indices(lo)<y%indices(right))
          right=right-1
        end do
        if(left>=right)exit
        call swap(left,right);left=left+1
      end do
      call partition_sort(left,hi,depth-1);call partition_sort(lo,left,depth-1)
    end subroutine
    subroutine sift(first,last,base)
      integer,intent(in)::first,last,base
      integer::root,child
      root=first
      do while(2*root<=last)
        child=2*root
        if(child<last)then
          if(y%indices(base+child-1)<=y%indices(base+child))child=child+1
        end if
        if(y%indices(base+root-1)>=y%indices(base+child-1))exit
        call swap(base+root-1,base+child-1);root=child
      end do
    end subroutine
  end function
  function sparse_rows(x,rows) result(y)
    type(csr_matrix),intent(in)::x
    integer,intent(in)::rows(:)
    type(csr_matrix)::y
    integer::i,a,b,offset
    y%n_rows=size(rows);y%n_cols=x%n_cols;allocate(y%indptr(size(rows)+1));y%indptr(1)=1
    do i=1,size(rows)
      y%indptr(i+1)=y%indptr(i)+x%indptr(rows(i)+1)-x%indptr(rows(i))
    end do
    allocate(y%indices(y%indptr(size(rows)+1)-1),y%data(y%indptr(size(rows)+1)-1))
    do i=1,size(rows)
      a=x%indptr(rows(i));b=x%indptr(rows(i)+1)-1;offset=y%indptr(i)
      y%indices(offset:offset+b-a)=x%indices(a:b);y%data(offset:offset+b-a)=x%data(a:b)
    end do
  end function
  function libsvm_sparse_gram(x,y,kernel,gamma,coef0,degree,training)result(k)
    type(csr_matrix),intent(in)::x,y
    character(*),intent(in)::kernel
    real(dp),intent(in)::gamma,coef0
    integer,intent(in)::degree
    logical,intent(in)::training
    real(dp),allocatable::k(:,:)
    real(dp)::dot,distance
    integer::i,j
    allocate(k(x%n_rows,y%n_rows))
    do j=1,y%n_rows
      do i=1,x%n_rows
        dot=sparse_inner(x,i,y,j,.false.)
        select case(kernel)
        case('linear');k(i,j)=dot
        case('poly','polynomial');k(i,j)=(gamma*dot+coef0)**degree
        case('sigmoid');k(i,j)=tanh(gamma*dot+coef0)
        case('rbf')
          if(training)then
            distance=sparse_inner(x,i,x,i,.false.)+sparse_inner(y,j,y,j,.false.)-2*dot
          else
            distance=sparse_inner(x,i,y,j,.true.)
          end if
          k(i,j)=exp(-gamma*distance)
        end select
      end do
    end do
  end function
end module
