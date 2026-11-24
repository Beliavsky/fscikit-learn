program nystroem_kernels
 use sklearn,only:dp,nystroem
 implicit none
 type(nystroem)::model
 real(dp)::x(5,3)
 real(dp),allocatable::z(:,:)
 integer::i
 do i=1,5
  x(i,:)=[real(i,dp),sin(real(i,dp)),cos(real(i,dp))]
 end do
 model%n_components=3;model%random_state=42;model%kernel='laplacian';model%gamma=.4_dp
 z=model%fit_transform(x)
 print *, 'One-based landmark rows:',model%component_indices_
 print *, 'First normalization row:',model%normalization_(1,:)
 print *, 'First feature row:',z(1,:)
 model%gamma=.8_dp;z=model%transform(x)
 print *, 'Live-gamma feature row:',z(1,:)
end program
