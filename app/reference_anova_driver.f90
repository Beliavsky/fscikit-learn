program reference_anova_driver
  use sklearn_common,only:dp
  use sklearn_univariate_selection,only:f_oneway,anova_sample,anova_result
  use sklearn_sparse,only:csr_matrix
  implicit none
  type(anova_sample),allocatable::samples(:)
  type(csr_matrix),allocatable::sparse(:)
  type(anova_result)::test
  real(dp),allocatable::x(:,:),a(:,:),b(:,:)
  integer::k,p,c,n,kind,nnz,i,status,route
  do
    read(*,*,iostat=status)k,p,route
    if(status/=0)exit
    allocate(samples(k),sparse(k))
    do c=1,k
      read(*,*)kind,n
      if(kind==0)then
        allocate(x(n,p))
        do i=1,n
          read(*,*)x(i,:)
        end do
        samples(c)=anova_sample(x)
        if(c==1)a=x
        if(c==2)b=x
        deallocate(x)
      else
        read(*,*)nnz
        sparse(c)%n_rows=n;sparse(c)%n_cols=p
        allocate(sparse(c)%indptr(n+1),sparse(c)%indices(nnz),sparse(c)%data(nnz))
        read(*,*)sparse(c)%indptr
        read(*,*)sparse(c)%indices
        read(*,*)sparse(c)%data
        samples(c)=anova_sample(sparse(c))
      end if
    end do
    select case(route)
    case(1)
      test=f_oneway(a,b)
    case(2)
      test=f_oneway(sparse(1),sparse(2))
    case default
      test=f_oneway(samples)
    end select
    write(*,'(*(es25.17e3,1x))')test%statistic
    write(*,'(*(es25.17e3,1x))')test%pvalue
    deallocate(samples,sparse)
  end do
end program
